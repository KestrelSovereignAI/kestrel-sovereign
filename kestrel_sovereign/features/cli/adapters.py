"""Feature-owned CLI adapters."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .terminal import (
    CliCommandDefinition,
    CliRisk,
    CliToolDeclaration,
    TerminalCommandRequest,
    TerminalExecutionService,
)


_GIT_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@~^+-]{0,255}$")
_GIT_ENV_KEYS = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SYSTEMROOT",
    "WINDIR",
)


class CliAdapterError(RuntimeError):
    """Raised when a registered CLI adapter command fails."""


@dataclass
class CliAvailabilityReport:
    adapter_id: str
    tools: dict[str, dict[str, Any]]

    @property
    def available(self) -> bool:
        return all(
            tool["available"] for tool in self.tools.values() if tool["required"]
        )


class FeatureCliAdapter:
    """Base contract for feature-specific CLI adapters."""

    adapter_id = "base"
    tools: tuple[CliToolDeclaration, ...] = ()
    commands: tuple[CliCommandDefinition, ...] = ()

    def __init__(self, terminal: TerminalExecutionService):
        self.terminal = terminal

    async def check_availability(self) -> CliAvailabilityReport:
        statuses: dict[str, dict[str, Any]] = {}
        for tool in self.tools:
            availability = await self.terminal.which(tool.name)
            statuses[tool.name] = {
                "required": tool.required,
                "available": availability.available,
                "path": availability.path,
                "version": availability.version,
            }
        return CliAvailabilityReport(adapter_id=self.adapter_id, tools=statuses)


class GitCliAdapter(FeatureCliAdapter):
    """Read-only local repository inspection through `git`."""

    adapter_id = "git"
    tools = (
        CliToolDeclaration("git", required=True),
    )
    commands = (
        CliCommandDefinition(
            "git.status",
            "Read local repository status using `git status --short --branch`.",
            CliRisk.READ_ONLY,
            ("git",),
        ),
        CliCommandDefinition(
            "git.diff",
            "Read local repository diff output.",
            CliRisk.READ_ONLY,
            ("git",),
        ),
        CliCommandDefinition(
            "git.log",
            "Read recent local repository commit history.",
            CliRisk.READ_ONLY,
            ("git",),
        ),
        CliCommandDefinition(
            "git.show_file",
            "Read a file from a local repository ref.",
            CliRisk.READ_ONLY,
            ("git",),
        ),
        CliCommandDefinition(
            "git.merge_base",
            "Read the merge-base for two local refs.",
            CliRisk.READ_ONLY,
            ("git",),
        ),
    )

    async def status(self, *, repo_path: str = ".") -> dict[str, Any]:
        result = await self._run_text(
            repo_path=repo_path,
            argv=["status", "--short", "--branch"],
            command_id="git.status",
        )
        return {
            "repo_path": str(_validate_local_repo_path(repo_path)),
            "status": result,
        }

    async def diff(
        self,
        *,
        repo_path: str = ".",
        ref: str = "",
        path: str = "",
    ) -> dict[str, Any]:
        argv = ["diff", "--no-ext-diff", "--no-textconv"]
        safe_ref = ""
        if ref:
            safe_ref = _validate_git_ref(ref, name="ref")
            argv.append(safe_ref)
        safe_path = ""
        if path:
            safe_path = _validate_git_pathspec(path)
            argv.extend(["--", safe_path])

        result = await self._run_text(
            repo_path=repo_path,
            argv=argv,
            command_id="git.diff",
        )
        return {
            "repo_path": str(_validate_local_repo_path(repo_path)),
            "ref": safe_ref,
            "path": safe_path,
            "diff": result,
        }

    async def log(
        self,
        *,
        repo_path: str = ".",
        max_count: int | str = 20,
    ) -> dict[str, Any]:
        count = _validate_positive_int(max_count, "max_count")
        result = await self._run_text(
            repo_path=repo_path,
            argv=[
                "log",
                "--oneline",
                "--decorate",
                "--max-count",
                str(min(count, 100)),
            ],
            command_id="git.log",
        )
        return {
            "repo_path": str(_validate_local_repo_path(repo_path)),
            "max_count": min(count, 100),
            "log": result,
        }

    async def show_file(
        self,
        *,
        repo_path: str = ".",
        ref: str = "HEAD",
        path: str,
    ) -> dict[str, Any]:
        safe_ref = _validate_git_ref(ref, name="ref")
        safe_path = _validate_git_pathspec(path)
        result = await self._run_text(
            repo_path=repo_path,
            argv=["show", f"{safe_ref}:{safe_path}"],
            command_id="git.show_file",
        )
        return {
            "repo_path": str(_validate_local_repo_path(repo_path)),
            "ref": safe_ref,
            "path": safe_path,
            "content": result,
        }

    async def merge_base(
        self,
        *,
        repo_path: str = ".",
        left_ref: str,
        right_ref: str,
    ) -> dict[str, Any]:
        safe_left = _validate_git_ref(left_ref, name="left_ref")
        safe_right = _validate_git_ref(right_ref, name="right_ref")
        result = await self._run_text(
            repo_path=repo_path,
            argv=["merge-base", safe_left, safe_right],
            command_id="git.merge_base",
        )
        return {
            "repo_path": str(_validate_local_repo_path(repo_path)),
            "left_ref": safe_left,
            "right_ref": safe_right,
            "merge_base": result.strip(),
        }

    async def _run_text(
        self,
        *,
        repo_path: str,
        argv: list[str],
        command_id: str,
    ) -> str:
        safe_repo_path = _validate_local_repo_path(repo_path)
        result = await self.terminal.run(
            TerminalCommandRequest(
                argv=["git", "--no-optional-locks", "-C", str(safe_repo_path), *argv],
                env=_git_env(),
                timeout=60,
                risk=CliRisk.READ_ONLY,
                command_id=command_id,
            )
        )
        if not result.ok:
            raise CliAdapterError(
                result.redacted_stderr or f"git command exited {result.returncode}"
            )
        if result.truncated_stdout or result.truncated_stderr:
            raise CliAdapterError("git command output exceeded the capture limit")
        return result.redacted_stdout


def _validate_positive_int(value: int | str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CliAdapterError(f"{name} must be a positive integer") from exc
    if parsed < 1 or str(value).strip() != str(parsed):
        raise CliAdapterError(f"{name} must be a positive integer")
    return parsed


def _validate_local_repo_path(repo_path: str) -> Path:
    if not isinstance(repo_path, str) or not repo_path.strip() or "\x00" in repo_path:
        raise CliAdapterError("repo_path must be a local directory")
    path = Path(repo_path).expanduser().resolve()
    if not path.is_dir():
        raise CliAdapterError("repo_path must be an existing local directory")
    allowed_roots = _allowed_git_repo_roots()
    if not any(_path_is_relative_to(path, root) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise CliAdapterError(f"repo_path must be under an allowed root: {roots}")
    return path


def _allowed_git_repo_roots() -> tuple[Path, ...]:
    roots = [Path.cwd().resolve()]
    configured = os.environ.get("KESTREL_CLI_ALLOWED_REPO_ROOTS", "")
    for raw_root in configured.split(os.pathsep):
        if raw_root.strip():
            roots.append(Path(raw_root).expanduser().resolve())
    return tuple(dict.fromkeys(roots))


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _git_env() -> dict[str, str]:
    env = {key: value for key in _GIT_ENV_KEYS if (value := os.environ.get(key))}
    env.update(
        {
            "GIT_EXTERNAL_DIFF": "",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return env


def _validate_git_ref(ref: str, *, name: str) -> str:
    if not isinstance(ref, str) or not ref.strip() or "\x00" in ref:
        raise CliAdapterError(f"{name} must be a non-empty git ref")
    safe_ref = ref.strip()
    if safe_ref.startswith("-") or not _GIT_REF_RE.fullmatch(safe_ref):
        raise CliAdapterError(f"{name} is not a safe git ref")
    if ".." in safe_ref:
        raise CliAdapterError(f"{name} must not contain revision ranges")
    return safe_ref


def _validate_git_pathspec(path: str) -> str:
    if not isinstance(path, str) or not path or path.startswith("/") or "\x00" in path:
        raise CliAdapterError("path must be a relative repository path")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise CliAdapterError("path must not contain empty, '.', or '..' segments")
    if path.startswith("-"):
        raise CliAdapterError("path must not look like a command option")
    return path
