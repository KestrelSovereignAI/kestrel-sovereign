"""CLI feature: registered feature-owned command-line adapters."""

from __future__ import annotations

import asyncio
from typing import Any

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.features.base import Feature, tool

from .adapters import CliAdapterError, GitCliAdapter
from .terminal import CliRisk, TerminalExecutionService, redact_secrets

_APPROVAL_TIMEOUT = 300
_SENSITIVE_OPTION_PARTS = (
    "api-key",
    "apikey",
    "auth",
    "credential",
    "key",
    "password",
    "passwd",
    "private-key",
    "pwd",
    "refresh-token",
    "secret",
    "token",
)


class CliFeature(Feature):
    """Expose feature-owned CLI adapters without opening arbitrary shell access."""

    tool_name = "cli"

    def __init__(self, agent=None):
        super().__init__(agent)
        self.terminal = TerminalExecutionService(
            approval_callback=self._approve_cli_command,
        )
        self.adapters = {
            "git": GitCliAdapter(self.terminal),
        }

    @property
    def tool_description(self) -> str:
        return (
            "Run registered, feature-owned CLI workflows using installed tools "
            "and the user's existing CLI authentication. This is not arbitrary shell access."
        )

    async def initialize(self) -> None:
        return None

    def _get_security_feature(self) -> Any:
        if hasattr(self.agent, "get_feature"):
            security = self.agent.get_feature("security")
            if security is not None:
                return security
            security = self.agent.get_feature("SecurityFeature")
            if security is not None:
                return security
        features = getattr(self.agent, "features", None)
        if isinstance(features, dict):
            return features.get("security") or features.get("SecurityFeature")
        return None

    async def _approve_cli_command(self, request: Any) -> bool:
        security = self._get_security_feature()
        queue = getattr(security, "approval_queue", None) if security else None
        if queue is None:
            return False

        try:
            approved, _scope = await queue.request_approval(
                feature_name="cli",
                tool_name=request.command_id or request.argv[0],
                tool_args={
                    "command_id": request.command_id,
                    "risk": (
                        request.risk.value
                        if isinstance(request.risk, CliRisk)
                        else str(request.risk)
                    ),
                    "argv": _approval_argv_summary(request.argv),
                    "argc": len(request.argv),
                    "cwd": str(request.cwd) if request.cwd else None,
                    "env_keys": sorted((request.env or {}).keys()),
                    "timeout": request.timeout,
                },
                timeout=_APPROVAL_TIMEOUT,
            )
        except (TimeoutError, asyncio.TimeoutError):
            return False
        except Exception:
            return False
        return bool(approved)

    @tool(
        name="cli_status",
        description="Show platform metadata and registered CLI adapter tool availability.",
        category=ToolCategory.SYSTEM,
        command_prefix="!cli-status",
    )
    async def cli_status(self) -> ToolResult:
        reports = {}
        for adapter_id, adapter in self.adapters.items():
            report = await adapter.check_availability()
            reports[adapter_id] = {
                "available": report.available,
                "tools": report.tools,
                "commands": [
                    {
                        "id": command.command_id,
                        "description": command.description,
                        "risk": command.risk.value,
                        "tools": list(command.tools),
                    }
                    for command in adapter.commands
                ],
            }
        return ToolResult.ok(
            "CLI adapter status collected.",
            data={
                "platform": self.terminal.platform_metadata(),
                "adapters": reports,
            },
        )

    @tool(
        name="git_status",
        description="Read local repository status via `git status --short --branch`.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!git-status",
    )
    async def git_status(self, repo_path: str = ".") -> ToolResult:
        try:
            payload = await self.adapters["git"].status(repo_path=repo_path)
        except (CliAdapterError, ValueError) as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read git status for {payload['repo_path']}.",
            data=payload,
        )

    @tool(
        name="git_diff",
        description="Read local repository diff via `git diff`.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!git-diff",
    )
    async def git_diff(
        self,
        ref: str = "",
        path: str = "",
        repo_path: str = ".",
    ) -> ToolResult:
        try:
            payload = await self.adapters["git"].diff(
                repo_path=repo_path,
                ref=ref,
                path=path,
            )
        except (CliAdapterError, ValueError) as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read git diff for {payload['repo_path']}.",
            data=payload,
        )

    @tool(
        name="git_log",
        description="Read recent local repository commits via `git log`.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!git-log",
    )
    async def git_log(self, max_count: int = 20, repo_path: str = ".") -> ToolResult:
        try:
            payload = await self.adapters["git"].log(
                repo_path=repo_path,
                max_count=max_count,
            )
        except (CliAdapterError, ValueError) as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read git log for {payload['repo_path']}.",
            data=payload,
        )

    @tool(
        name="git_show_file",
        description="Read a local repository file from a git ref via `git show`.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!git-show-file",
    )
    async def git_show_file(
        self,
        ref: str,
        path: str,
        repo_path: str = ".",
    ) -> ToolResult:
        try:
            payload = await self.adapters["git"].show_file(
                repo_path=repo_path,
                ref=ref,
                path=path,
            )
        except (CliAdapterError, ValueError) as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read {path} at {ref} from {payload['repo_path']}.",
            data=payload,
        )

    @tool(
        name="git_merge_base",
        description="Read the merge-base for two local git refs.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!git-merge-base",
    )
    async def git_merge_base(
        self,
        left_ref: str,
        right_ref: str,
        repo_path: str = ".",
    ) -> ToolResult:
        try:
            payload = await self.adapters["git"].merge_base(
                repo_path=repo_path,
                left_ref=left_ref,
                right_ref=right_ref,
            )
        except (CliAdapterError, ValueError) as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read git merge-base for {payload['repo_path']}.",
            data=payload,
        )


def _command_data(result: Any) -> dict[str, Any]:
    return {
        "argv": result.argv,
        "risk": CliRisk.READ_ONLY.value,
        "returncode": result.returncode,
        "stdout": result.redacted_stdout,
        "stderr": result.redacted_stderr,
        "duration_ms": result.duration_ms,
        "timed_out": result.timed_out,
    }


def _approval_argv_summary(argv: list[Any]) -> list[str]:
    """Return an approval-safe argv preview without positional values."""

    summary: list[str] = []
    redact_next = False
    for index, raw_arg in enumerate(argv):
        arg = redact_secrets(str(raw_arg))
        if index == 0:
            summary.append(arg)
            continue
        if redact_next:
            summary.append("[REDACTED]")
            redact_next = False
            continue
        if arg.startswith("--"):
            name, separator, _value = arg.partition("=")
            if _is_sensitive_option(name):
                if separator:
                    summary.append(f"{name}=[REDACTED]")
                else:
                    summary.append(name)
                    redact_next = True
            elif separator:
                summary.append(f"{name}=[ARG]")
            else:
                summary.append(name)
            continue
        if arg.startswith("-") and len(arg) > 1:
            summary.append(_short_option_summary(arg))
            if _is_sensitive_option(arg):
                redact_next = True
            continue
        summary.append("[ARG]")
    return summary


def _is_sensitive_option(option: str) -> bool:
    normalized = option.lstrip("-").lower().replace("_", "-")
    return any(part in normalized for part in _SENSITIVE_OPTION_PARTS)


def _short_option_summary(option: str) -> str:
    if len(option) <= 2:
        return option
    if "=" in option:
        return f"{option[:2]}=[ARG]"
    return f"{option[:2]}[ARG]"
