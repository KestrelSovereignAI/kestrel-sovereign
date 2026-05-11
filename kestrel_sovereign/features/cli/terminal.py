"""Shared terminal execution substrate for feature-owned CLI adapters."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import platform
import re
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable


_TOKEN_PATTERNS = [
    re.compile(r"(gh[opsu]_[A-Za-z0-9_]{20,})"),
    re.compile(r"((?:sk|xox[baprs]|github_pat)_[A-Za-z0-9_\-]{20,})"),
    re.compile(r"(sk-[A-Za-z0-9][A-Za-z0-9_\-]{10,})"),
    re.compile(r"(?i)(token|api[_-]?key|secret|password)(\s*[:=]\s*)([^\s]+)"),
]


class CliRisk(str, Enum):
    """Risk classification for registered CLI commands."""

    READ_ONLY = "read_only"
    LOCAL_MUTATION = "local_mutation"
    REMOTE_MUTATION = "remote_mutation"
    DESTRUCTIVE = "destructive"
    CREDENTIAL_AFFECTING = "credential_affecting"
    EXTERNAL_TRANSMISSION = "external_transmission"
    FINANCIAL_OR_BILLING = "financial_or_billing"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ToolAvailability:
    """Resolved executable metadata."""

    name: str
    path: str | None
    available: bool
    version: str | None = None


@dataclass(frozen=True)
class TerminalCommandRequest:
    """A command request built by a feature CLI adapter."""

    argv: list[str]
    cwd: Path | None = None
    env: dict[str, str] | None = None
    timeout: int = 60
    risk: CliRisk = CliRisk.UNKNOWN
    command_id: str = ""


@dataclass(frozen=True)
class TerminalCommandResult:
    """Captured command result with redacted stdout/stderr."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    truncated_stdout: bool = False
    truncated_stderr: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def json_stdout(self) -> Any:
        return json.loads(self.stdout or "null")

    @property
    def redacted_stdout(self) -> str:
        return redact_secrets(self.stdout)

    @property
    def redacted_stderr(self) -> str:
        return redact_secrets(self.stderr)


@dataclass(frozen=True)
class CliToolDeclaration:
    """Executable declaration owned by a feature adapter."""

    name: str
    required: bool = True
    version_args: list[str] = field(default_factory=lambda: ["--version"])


@dataclass(frozen=True)
class CliCommandDefinition:
    """Registered command metadata exposed by an adapter."""

    command_id: str
    description: str
    risk: CliRisk
    tools: tuple[str, ...]


CliApprovalCallback = Callable[[TerminalCommandRequest], Awaitable[bool]]


class TerminalExecutionService:
    """Run argument-vector commands without shell interpolation."""

    def __init__(
        self,
        *,
        max_output_bytes: int = 1_048_576,
        approval_callback: CliApprovalCallback | None = None,
    ):
        self.max_output_bytes = max_output_bytes
        self.approval_callback = approval_callback

    def platform_metadata(self) -> dict[str, str]:
        return {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        }

    async def which(self, command: str) -> ToolAvailability:
        path = shutil.which(command)
        version = None
        if path:
            result = await self.run(
                TerminalCommandRequest(
                    argv=[command, "--version"],
                    timeout=10,
                    risk=CliRisk.READ_ONLY,
                    command_id=f"{command}.version",
                )
            )
            version_output = result.stdout or result.stderr
            version = version_output.splitlines()[0] if version_output else None
        return ToolAvailability(
            name=command,
            path=path,
            available=path is not None,
            version=version,
        )

    async def run(self, request: TerminalCommandRequest) -> TerminalCommandResult:
        if not request.argv:
            raise ValueError("argv must not be empty")

        started = time.monotonic()
        approval_error = await self._approval_error(request)
        if approval_error:
            return TerminalCommandResult(
                argv=list(request.argv),
                returncode=126,
                stdout="",
                stderr=approval_error,
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                *request.argv,
                cwd=str(request.cwd) if request.cwd else None,
                env=request.env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            return TerminalCommandResult(
                argv=list(request.argv),
                returncode=127,
                stdout="",
                stderr=str(exc),
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        timed_out = False
        try:
            stdout_task = asyncio.create_task(self._read_limited(proc.stdout, proc))
            stderr_task = asyncio.create_task(self._read_limited(proc.stderr, proc))
            wait_task = asyncio.create_task(proc.wait())
            gather_task = asyncio.gather(stdout_task, stderr_task, wait_task)
            (stdout_bytes, stdout_truncated), (stderr_bytes, stderr_truncated), _ = (
                await asyncio.wait_for(
                    asyncio.shield(gather_task),
                    timeout=request.timeout,
                )
            )
        except asyncio.TimeoutError:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            stdout_bytes, stdout_truncated = await stdout_task
            stderr_bytes, stderr_truncated = await stderr_task
            await proc.wait()

        stdout = self._decode_output(stdout_bytes)
        stderr = self._decode_output(stderr_bytes)
        return TerminalCommandResult(
            argv=list(request.argv),
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr,
            duration_ms=int((time.monotonic() - started) * 1000),
            timed_out=timed_out,
            truncated_stdout=stdout_truncated,
            truncated_stderr=stderr_truncated,
        )

    async def _read_limited(
        self,
        stream: asyncio.StreamReader | None,
        proc: asyncio.subprocess.Process,
    ) -> tuple[bytes, bool]:
        if stream is None:
            return b"", False

        data = bytearray()
        truncated = False
        while True:
            chunk = await stream.read(65_536)
            if not chunk:
                break

            remaining = self.max_output_bytes - len(data)
            if remaining > 0:
                data.extend(chunk[:remaining])

            if len(chunk) > remaining:
                truncated = True
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                break
        return bytes(data), truncated

    def _decode_output(self, data: bytes) -> str:
        return data.decode("utf-8", errors="replace")

    async def _approval_error(self, request: TerminalCommandRequest) -> str | None:
        if request.risk == CliRisk.READ_ONLY:
            return None

        risk = request.risk.value if isinstance(request.risk, CliRisk) else str(request.risk)
        if self.approval_callback is None:
            return (
                f"CLI command {request.command_id or request.argv[0]} has risk "
                f"{risk} and requires approval, but no approval callback is configured"
            )

        try:
            approved = await self.approval_callback(request)
        except Exception as exc:  # noqa: BLE001
            return redact_secrets(f"CLI command approval failed for risk {risk}: {exc}")
        if not approved:
            return f"CLI command denied by approval gate for risk {risk}"
        return None


def redact_secrets(value: str) -> str:
    """Redact common token forms from command output."""

    redacted = value
    for pattern in _TOKEN_PATTERNS:
        if pattern.groups >= 3:
            redacted = pattern.sub(r"\1\2[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def decode_github_content_response(payload: dict[str, Any]) -> str:
    """Decode a `gh api /contents` response body."""

    content = payload.get("content")
    encoding = payload.get("encoding")
    if not isinstance(content, str) or encoding != "base64":
        raise ValueError("GitHub contents response is not base64 encoded")
    compact = "".join(content.splitlines())
    return base64.b64decode(compact).decode("utf-8", errors="replace")
