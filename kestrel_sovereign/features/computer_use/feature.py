"""ComputerUseFeature: bounded host access wrapped in three gates.

Gate order on every tool call:

1. **Privacy** — ``PrivacyConfig.computer_access`` must be ``True``.
2. **Constitution** — the operation's capability name must be present in
   the agent's Amendment IX grants.
3. **Approval** — the per-call decision from
   :class:`SecurityFeature.approval_queue`.

A failure at any gate produces a structured ``denied`` result that the
agent loop can surface to the user. Every call — allowed, denied, or
errored — produces a row in the JSONL audit log.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sovereign.constitution.hierarchy import (
    DANGEROUS_CAPABILITIES,
    parse_amendment_ix_grants,
)
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

from .audit import AuditLog, AuditRecord
from .backends import (
    CapabilityBlocked,
    DockerSandboxBackend,
    LocalSandboxBackend,
    SandboxBackend,
)
from .path_safety import PathSafetyError, resolve_against_roots
from .policy import BinaryPolicy, Decision, PathPolicy

logger = logging.getLogger(__name__)


_DEFAULT_DENY_PATHS = ["~/.ssh", "~/.aws", "~/.config", "~/.gnupg"]
_DEFAULT_ALLOWED_BINS = ["git", "ls", "cat", "rg", "uv", "node", "python"]
_DEFAULT_DENIED_BINS = ["rm", "dd", "mkfs", "shutdown", "sudo", "ssh"]
_APPROVAL_TIMEOUT = 300.0


class ComputerUseFeature(Feature):
    """Bounded host access (filesystem + shell) for the sovereign agent."""

    tool_name = "computer_use"

    def __init__(self, agent=None):
        super().__init__(agent)
        self._cfg: Dict[str, Any] = {}
        self._backend: Optional[SandboxBackend] = None
        self._path_policy: Optional[PathPolicy] = None
        self._binary_policy: Optional[BinaryPolicy] = None
        self._audit: Optional[AuditLog] = None
        self._allowed_roots: List[Path] = []
        self._max_read_bytes: int = 5_000_000

    @property
    def tool_description(self) -> str:
        return (
            "Bounded host access: read/list files in the allow-list, "
            "approval-gated writes/edits, and shell execution. "
            "Disabled unless privacy.computer_access, Amendment IX, and "
            "[features.computer_use] are all set."
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def initialize(self) -> None:
        # Pre-populated _cfg (e.g. from a test) takes precedence over toml.
        if not self._cfg:
            self._cfg = self._load_config()
        await self._setup_from_config()

    async def _setup_from_config(self) -> None:
        if not self._cfg.get("enabled", False):
            logger.info("ComputerUseFeature: disabled in kestrel.toml")
            return

        allowed = [Path(p).expanduser() for p in self._cfg.get("allowed_paths", [])]
        denied = [str(Path(p).expanduser()) for p in self._cfg.get("deny_paths", _DEFAULT_DENY_PATHS)]
        self._allowed_roots = allowed
        self._max_read_bytes = int(self._cfg.get("max_read_bytes", 5_000_000))

        self._path_policy = PathPolicy(
            allow=[str(p) for p in allowed],
            deny=denied,
            auto_approve_read=bool(self._cfg.get("auto_approve_read", True)),
        )
        self._binary_policy = BinaryPolicy(
            allow=list(self._cfg.get("allowed_binaries", _DEFAULT_ALLOWED_BINS)),
            deny=list(self._cfg.get("denied_binaries", _DEFAULT_DENIED_BINS)),
        )

        audit_path = self._cfg.get("audit_log_path", ".kestrel/computer_use_audit.jsonl")
        self._audit = AuditLog(audit_path, agent=self.agent)

        granted = self._granted_capabilities()
        backend_name = self._cfg.get("backend", "docker")
        try:
            if backend_name == "local":
                self._backend = LocalSandboxBackend(granted)
            else:
                self._backend = DockerSandboxBackend(
                    granted_capabilities=granted,
                    memory_limit=self._cfg.get("docker", {}).get("memory_limit", "256m"),
                )
        except CapabilityBlocked as exc:
            logger.warning("ComputerUseFeature: backend refused init: %s", exc)
            self._backend = None
            return

        logger.info(
            "ComputerUseFeature initialized: backend=%s, allowed_roots=%d, allowed_bins=%d",
            self._backend.name,
            len(self._allowed_roots),
            len(self._binary_policy.allow),
        )

    async def shutdown(self) -> None:
        if self._backend is not None:
            await self._backend.shutdown()

    def _load_config(self) -> Dict[str, Any]:
        """Read [features.computer_use] from kestrel.toml. Best-effort."""
        try:
            try:
                import tomllib  # type: ignore[import-not-found]
            except ImportError:
                import tomli as tomllib  # type: ignore[import-not-found]
        except Exception:
            return {}

        # Walk up from this file to find kestrel.toml at the repo root.
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "kestrel.toml"
            if candidate.exists():
                try:
                    with open(candidate, "rb") as f:
                        data = tomllib.load(f)
                    return data.get("features", {}).get("computer_use", {}) or {}
                except Exception as exc:  # noqa: BLE001
                    logger.warning("failed to read kestrel.toml: %s", exc)
                    return {}
        return {}

    def _granted_capabilities(self) -> frozenset[str]:
        """Read Amendment IX grants from the agent's constitution.

        Resolution order:

        1. ``self.agent.granted_capabilities`` if the agent exposes it
           (host applications can pre-resolve grants once at start-up).
        2. Parse Amendment IX out of the agent's constitution text. The
           agent may expose ``constitution_text``; otherwise we read the
           on-disk ``KESTREL_CONSTITUTION.md`` referenced by ``config``.
        3. ``KESTREL_CAPABILITY_GRANTS`` env var (test-only convenience).

        Empty set means *no grants* — every dangerous tool is refused at
        the constitutional gate, which is the safe default.
        """
        for attr in ("granted_capabilities", "capability_grants", "amendment_ix"):
            grants = getattr(self.agent, attr, None)
            if grants:
                return frozenset(grants)

        text = getattr(self.agent, "constitution_text", None)
        if not text:
            text = self._read_constitution_text()
        if text:
            grants = parse_amendment_ix_grants(text)
            if grants:
                return grants

        env_grants = os.environ.get("KESTREL_CAPABILITY_GRANTS", "")
        if env_grants:
            return frozenset(g.strip() for g in env_grants.split(",") if g.strip())
        return frozenset()

    def _read_constitution_text(self) -> str | None:
        """Load the canonical ``KESTREL_CONSTITUTION.md`` from disk."""
        try:
            from kestrel_sovereign.config import CONSTITUTION_PATH

            return Path(CONSTITUTION_PATH).read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not read constitution: %s", exc)
            return None

    # =========================================================================
    # Gate logic
    # =========================================================================

    def _privacy_allows(self) -> bool:
        cfg = getattr(self.agent, "privacy_config", None)
        if cfg is None:
            return False
        getter = getattr(cfg, "allows_computer_access", None)
        return bool(getter()) if callable(getter) else bool(getattr(cfg, "computer_access", False))

    def _constitution_allows(self, capability: str) -> bool:
        if capability not in DANGEROUS_CAPABILITIES:
            return True
        return capability in self._granted_capabilities()

    def _get_security_feature(self):
        if hasattr(self.agent, "get_feature"):
            return self.agent.get_feature("security")
        if hasattr(self.agent, "features"):
            return self.agent.features.get("security")
        return None

    async def _request_approval(
        self,
        action: str,
        payload: Dict[str, Any],
    ) -> tuple[bool, str]:
        security = self._get_security_feature()
        queue = getattr(security, "approval_queue", None) if security else None
        if queue is None:
            logger.warning("ComputerUseFeature: SecurityFeature unavailable; refusing %s", action)
            return False, "no_security_feature"
        try:
            approved, scope = await queue.request_approval(
                feature_name="computer_use",
                tool_name=action,
                tool_args=payload,
                timeout=_APPROVAL_TIMEOUT,
            )
            return approved, scope
        except (TimeoutError, asyncio.TimeoutError):
            return False, "timeout"
        except Exception as exc:  # noqa: BLE001
            logger.error("approval request failed: %s", exc, exc_info=True)
            return False, "error"

    async def _check_gates_or_audit(
        self,
        *,
        tool_name: str,
        capability: str,
        payload: Dict[str, Any],
        require_approval: bool,
    ) -> tuple[bool, List[str], str]:
        """Run the three gates. Returns (allowed, allowed_by, denied_reason)."""
        allowed_by: List[str] = []

        if not self._privacy_allows():
            await self._audit_denied(tool_name, payload, "privacy")
            return False, allowed_by, "privacy:computer_access flag is False"
        allowed_by.append("privacy")

        if not self._constitution_allows(capability):
            await self._audit_denied(tool_name, payload, "constitution")
            return False, allowed_by, f"constitution:Amendment IX missing grant '{capability}'"
        allowed_by.append("constitution")

        if require_approval:
            approved, scope = await self._request_approval(tool_name, payload)
            if not approved:
                await self._audit_denied(tool_name, payload, f"approval:{scope}")
                return False, allowed_by, f"approval:{scope}"
            allowed_by.append(f"approval:{scope}")

        return True, allowed_by, ""

    async def _audit_denied(
        self, tool_name: str, payload: Dict[str, Any], gate: str
    ) -> None:
        if self._audit is None or self._backend is None:
            return
        await self._audit.write(
            AuditRecord(
                tool=tool_name,
                backend=self._backend.name,
                args=payload,
                allowed_by=[gate],
                outcome="denied",
                agent_did=getattr(self.agent, "did", "anonymous"),
            )
        )

    async def _audit_run(
        self,
        *,
        tool_name: str,
        payload: Dict[str, Any],
        allowed_by: List[str],
        outcome: str,
        duration_ms: int,
        error: Optional[str] = None,
    ) -> None:
        if self._audit is None or self._backend is None:
            return
        await self._audit.write(
            AuditRecord(
                tool=tool_name,
                backend=self._backend.name,
                args=payload,
                allowed_by=allowed_by,
                outcome=outcome,
                duration_ms=duration_ms,
                error=error,
                agent_did=getattr(self.agent, "did", "anonymous"),
            )
        )

    def _resolve(self, path: str) -> Path:
        if not self._allowed_roots:
            raise PathSafetyError("no allowed_paths configured")
        return resolve_against_roots(self._allowed_roots, path)

    def _ready_or_error(self) -> Optional[Dict[str, Any]]:
        if not self._cfg.get("enabled", False):
            return {"success": False, "error": "computer-use feature not enabled in kestrel.toml"}
        if self._backend is None:
            return {"success": False, "error": "computer-use backend failed to initialize"}
        return None

    # =========================================================================
    # Tools
    # =========================================================================

    @tool(
        name="fs_read",
        description="Read a file from the allow-list paths.",
        category=ToolCategory.FILE_OPERATIONS,
        command_prefix="!fs-read",
    )
    async def fs_read(self, path: str) -> Dict[str, Any]:
        """Read a file the sovereign has authorized.

        Args:
            path: Absolute or allow-list-relative path to read.
        """
        not_ready = self._ready_or_error()
        if not_ready:
            return not_ready

        try:
            resolved = self._resolve(path)
        except PathSafetyError as exc:
            return {"success": False, "error": f"path_safety:{exc}"}

        decision = self._path_policy.evaluate(resolved, write=False)
        if decision.decision is Decision.DENY:
            return {"success": False, "error": f"policy:{decision.rule}"}
        require_approval = decision.decision is Decision.REQUIRE_APPROVAL

        payload = {"path": str(resolved), "rule": decision.rule}
        ok, allowed_by, reason = await self._check_gates_or_audit(
            tool_name="fs-read",
            capability="filesystem_read",
            payload=payload,
            require_approval=require_approval,
        )
        if not ok:
            return {"success": False, "error": reason}

        import time as _t

        started = _t.monotonic()
        try:
            data = await self._backend.read(resolved, max_bytes=self._max_read_bytes)
            duration_ms = int((_t.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="fs-read",
                payload={**payload, "bytes": len(data)},
                allowed_by=allowed_by,
                outcome="ok",
                duration_ms=duration_ms,
            )
            return {
                "success": True,
                "path": str(resolved),
                "bytes": len(data),
                "content": data.decode("utf-8", errors="replace"),
            }
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((_t.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="fs-read",
                payload=payload,
                allowed_by=allowed_by,
                outcome="error",
                duration_ms=duration_ms,
                error=str(exc),
            )
            return {"success": False, "error": str(exc)}

    @tool(
        name="fs_list",
        description="List directory contents within the allow-list.",
        category=ToolCategory.FILE_OPERATIONS,
        command_prefix="!fs-list",
    )
    async def fs_list(self, path: str) -> Dict[str, Any]:
        """List a directory the sovereign has authorized.

        Args:
            path: Absolute or allow-list-relative directory path.
        """
        not_ready = self._ready_or_error()
        if not_ready:
            return not_ready

        try:
            resolved = self._resolve(path)
        except PathSafetyError as exc:
            return {"success": False, "error": f"path_safety:{exc}"}

        decision = self._path_policy.evaluate(resolved, write=False)
        if decision.decision is Decision.DENY:
            return {"success": False, "error": f"policy:{decision.rule}"}
        require_approval = decision.decision is Decision.REQUIRE_APPROVAL

        payload = {"path": str(resolved), "rule": decision.rule}
        ok, allowed_by, reason = await self._check_gates_or_audit(
            tool_name="fs-list",
            capability="filesystem_read",
            payload=payload,
            require_approval=require_approval,
        )
        if not ok:
            return {"success": False, "error": reason}

        import time as _t

        started = _t.monotonic()
        try:
            entries = await self._backend.list(resolved)
            duration_ms = int((_t.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="fs-list",
                payload={**payload, "count": len(entries)},
                allowed_by=allowed_by,
                outcome="ok",
                duration_ms=duration_ms,
            )
            return {
                "success": True,
                "path": str(resolved),
                "entries": [e.__dict__ for e in entries],
            }
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((_t.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="fs-list",
                payload=payload,
                allowed_by=allowed_by,
                outcome="error",
                duration_ms=duration_ms,
                error=str(exc),
            )
            return {"success": False, "error": str(exc)}

    @tool(
        name="fs_write",
        description="Write bytes to a file within the allow-list (always approval-gated).",
        category=ToolCategory.FILE_OPERATIONS,
        command_prefix="!fs-write",
    )
    async def fs_write(self, path: str, content: str) -> Dict[str, Any]:
        """Write to a file (always approval-gated).

        Args:
            path: Absolute or allow-list-relative path to write.
            content: UTF-8 content to write.
        """
        not_ready = self._ready_or_error()
        if not_ready:
            return not_ready

        try:
            resolved = self._resolve(path)
        except PathSafetyError as exc:
            return {"success": False, "error": f"path_safety:{exc}"}

        decision = self._path_policy.evaluate(resolved, write=True)
        if decision.decision is Decision.DENY:
            return {"success": False, "error": f"policy:{decision.rule}"}

        new_bytes = content.encode("utf-8")
        diff_preview = _diff_preview(resolved, new_bytes)
        payload = {
            "path": str(resolved),
            "bytes": len(new_bytes),
            "rule": decision.rule,
            "diff_preview": diff_preview,
        }
        ok, allowed_by, reason = await self._check_gates_or_audit(
            tool_name="fs-write",
            capability="filesystem_write",
            payload=payload,
            require_approval=True,
        )
        if not ok:
            return {"success": False, "error": reason}

        import time as _t

        started = _t.monotonic()
        try:
            written = await self._backend.write(resolved, new_bytes)
            duration_ms = int((_t.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="fs-write",
                payload={"path": str(resolved), "bytes": written, "rule": decision.rule},
                allowed_by=allowed_by,
                outcome="ok",
                duration_ms=duration_ms,
            )
            return {"success": True, "path": str(resolved), "bytes_written": written}
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((_t.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="fs-write",
                payload={"path": str(resolved), "rule": decision.rule},
                allowed_by=allowed_by,
                outcome="error",
                duration_ms=duration_ms,
                error=str(exc),
            )
            return {"success": False, "error": str(exc)}

    @tool(
        name="shell",
        description="Run a shell command within the configured backend (always approval-gated).",
        category=ToolCategory.SYSTEM,
        command_prefix="!shell",
    )
    async def shell(self, command: str, timeout: int = 60) -> Dict[str, Any]:
        """Run a shell command after policy + approval.

        Args:
            command: The shell command to run; tokenized with shlex.
            timeout: Wall-clock seconds before the process is killed.
        """
        not_ready = self._ready_or_error()
        if not_ready:
            return not_ready

        from .policy import split_command

        argv = split_command(command)
        if not argv:
            return {"success": False, "error": "empty command"}

        decision = self._binary_policy.evaluate(argv)
        if decision.decision is Decision.DENY:
            return {"success": False, "error": f"policy:{decision.rule}"}

        capability = (
            "shell_execution_host"
            if self._backend.name == "local"
            else "shell_execution_sandboxed"
        )
        payload = {
            "argv": argv,
            "binary": Path(argv[0]).name,
            "rule": decision.rule,
            "backend": self._backend.name,
            "timeout": timeout,
        }
        ok, allowed_by, reason = await self._check_gates_or_audit(
            tool_name="shell",
            capability=capability,
            payload=payload,
            require_approval=True,
        )
        if not ok:
            return {"success": False, "error": reason}

        import time as _t

        started = _t.monotonic()
        try:
            result = await self._backend.exec(argv, cwd=None, env=None, timeout=timeout)
            duration_ms = int((_t.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="shell",
                payload={**payload, "returncode": result.returncode},
                allowed_by=allowed_by,
                outcome="ok" if result.returncode == 0 else "error",
                duration_ms=duration_ms,
                error=None if result.returncode == 0 else f"exit {result.returncode}",
            )
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_ms": result.duration_ms,
                "timed_out": result.timed_out,
            }
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((_t.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="shell",
                payload=payload,
                allowed_by=allowed_by,
                outcome="error",
                duration_ms=duration_ms,
                error=str(exc),
            )
            return {"success": False, "error": str(exc)}


def _diff_preview(path: Path, new_bytes: bytes, *, max_chars: int = 4000) -> str:
    """Best-effort textual preview of what fs-write will change.

    Returns a small unified-diff snippet when both old and new content are
    valid UTF-8; otherwise reports byte-length deltas. Never raises.
    """
    try:
        old = path.read_bytes() if path.exists() else b""
    except OSError:
        old = b""

    try:
        old_text = old.decode("utf-8")
        new_text = new_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return f"binary write: {len(old)} -> {len(new_bytes)} bytes"

    import difflib

    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=str(path) + "@old",
        tofile=str(path) + "@new",
        n=2,
    )
    rendered = "".join(diff)
    if len(rendered) > max_chars:
        return rendered[:max_chars] + f"\n... [truncated {len(rendered) - max_chars} chars]"
    return rendered or "(no textual diff)"
