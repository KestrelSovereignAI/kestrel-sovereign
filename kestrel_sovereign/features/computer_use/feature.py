"""ComputerUseFeature: bounded host access wrapped in three gates + policy.

Every tool call runs through the same sequence:

1. **Readiness** — feature enabled in ``kestrel.toml``, backend constructed.
2. **Privacy** — ``PrivacyConfig.computer_access`` must be ``True``.
3. **Constitution** — the operation's capability name must be present in
   the agent's Amendment IX grants.
4. **Path safety** — only structural checks (``..`` traversal, NUL bytes).
   Allow-list / deny-list membership is **not** a safety concern; it is
   a policy decision that may legitimately produce ``REQUIRE_APPROVAL``.
5. **Policy** — evaluate the resolved realpath / argv against the
   allow + deny lists. ``DENY`` short-circuits before approval; ``ALLOW``
   skips approval (reads only); ``REQUIRE_APPROVAL`` forwards.
6. **Approval** — per-call human approval through
   :class:`SecurityFeature.approval_queue`.

Privacy and constitution come first because they are call-level and
input-independent — they tell us whether the call is *eligible* at all,
without needing to look at the path or the binary. Path-safety and policy
are input validation; they only matter once the call is eligible.

Every call — allowed, denied at any stage, or errored during execution —
produces exactly one row in the JSONL audit log. The audit row's
``allowed_by`` field accumulates the gates that passed; on a denial it
ends with ``denied:<gate>``. Reading those rows back is the canonical way
to reconstruct what happened.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from kestrel_sovereign.constitution.hierarchy import (
    DANGEROUS_CAPABILITIES,
    parse_amendment_ix_grants,
)
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.result import ToolResult
from kestrel_sdk.tools.base import ToolCategory

from .audit import AuditLog, AuditRecord
from .backends import (
    CapabilityBlocked,
    DockerSandboxBackend,
    LocalSandboxBackend,
    SandboxBackend,
)
from .path_safety import PathSafetyError, resolve_realpath
from .policy import (
    BinaryPolicy,
    Decision,
    PathPolicy,
    PolicyResult,
    command_contains_unquoted_shell_control,
    split_command,
)

logger = logging.getLogger(__name__)


_DEFAULT_DENY_PATHS = ["~/.ssh", "~/.aws", "~/.config", "~/.gnupg"]
# Auto-approved binaries skip the ApprovalQueue (#1694). The default
# list is deliberately narrow: only inert read-only tools where the
# argv shape can't be turned into arbitrary host work via flags or
# args. Interpreters (``python``, ``node``), package managers (``uv``,
# ``gh``) and other rich CLIs are NOT auto-approved by default — they
# still route through the queue so the operator (or a scoped
# auto-approve rule in security.permission_store) decides. Operators
# who want to broaden the default add binaries to
# ``[features.computer_use].auto_approved_binaries`` in kestrel.toml.
#
# The bridge handler additionally downgrades ALLOW to REQUIRE_APPROVAL
# when the raw command string contains shell metacharacters
# (``;``, ``&&``, ``||``, ``|``, backticks, ``$(...)``, redirects,
# newlines) so an allow-listed first token can never bless a piggy-
# backed compound command.
_DEFAULT_AUTO_APPROVED_BINS = ["ls", "cat", "rg"]
_DEFAULT_DENIED_BINS = ["rm", "dd", "mkfs", "shutdown", "sudo", "ssh"]
_APPROVAL_TIMEOUT = 300.0


@dataclass
class _GateOutcome:
    """Result of running the gate sequence for a single call."""

    allowed: bool
    allowed_by: List[str]
    denied_reason: str = ""

    def __bool__(self) -> bool:  # convenience for ``if outcome:``
        return self.allowed


class _PreApprovalRefusal(Exception):
    """Raised inside a pre-approval hook to refuse a call cleanly.

    The ``reason`` is appended to the audit chain as
    ``denied:input_validation:<reason>`` and surfaced to the caller as
    ``input_validation:<reason>``. Use this for input-dependent checks
    that must run *after* path-safety + policy have authorized the path
    but before the human approver sees the request — e.g. file-decode
    failures, missing match text, parameter validation that depends on
    file contents.
    """

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(message)


_PreApprovalHook = Callable[[Path, Dict[str, Any]], Awaitable[Dict[str, Any]]]


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
        self._max_read_bytes: int = 5_000_000

    @property
    def tool_description(self) -> str:
        return (
            "Bounded host access: read/list files, approval-gated writes/edits, "
            "and shell execution. Disabled unless privacy.computer_access, "
            "Amendment IX, and [features.computer_use] are all configured."
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
            # Audit log still opens so disabled-call refusals can be recorded.
            audit_path = self._cfg.get(
                "audit_log_path", ".kestrel/computer_use_audit.jsonl"
            )
            self._audit = AuditLog(audit_path, agent=self.agent)
            return

        allowed = [str(Path(p).expanduser()) for p in self._cfg.get("allowed_paths", [])]
        denied = [str(Path(p).expanduser()) for p in self._cfg.get("deny_paths", _DEFAULT_DENY_PATHS)]
        self._max_read_bytes = int(self._cfg.get("max_read_bytes", 5_000_000))

        self._path_policy = PathPolicy(
            allow=allowed,
            deny=denied,
            auto_approve_read=bool(self._cfg.get("auto_approve_read", True)),
        )
        # `auto_approved_binaries` is the canonical key (#1694). The
        # `allowed_binaries` spelling is accepted as a one-release
        # deprecation synonym: if `auto_approved_binaries` is set it
        # wins outright; if only `allowed_binaries` is set we honor it
        # and log a one-time WARNING; if both are set the canonical key
        # wins and we WARN that the legacy one is being ignored.
        cfg_auto = self._cfg.get("auto_approved_binaries")
        cfg_legacy = self._cfg.get("allowed_binaries")
        if cfg_auto is not None:
            auto_bins = list(cfg_auto)
            if cfg_legacy is not None:
                logger.warning(
                    "ComputerUseFeature: both `auto_approved_binaries` and "
                    "legacy `allowed_binaries` set in [features.computer_use]; "
                    "ignoring `allowed_binaries`. Remove it from kestrel.toml."
                )
        elif cfg_legacy is not None:
            logger.warning(
                "ComputerUseFeature: `allowed_binaries` is a deprecated "
                "alias for `auto_approved_binaries`. Rename it in "
                "kestrel.toml. The legacy spelling will be removed in a "
                "future release."
            )
            auto_bins = list(cfg_legacy)
        else:
            auto_bins = list(_DEFAULT_AUTO_APPROVED_BINS)
        self._binary_policy = BinaryPolicy(
            allow=auto_bins,
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
            "ComputerUseFeature initialized: backend=%s, allowed=%d, "
            "auto_approved_bins=%d, denied_bins=%d",
            self._backend.name,
            len(allowed),
            len(self._binary_policy.allow),
            len(self._binary_policy.deny),
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

        # Look for kestrel.toml in (1) the agent's storage dir, (2) the
        # agent_data/<name>/ root if that's the storage layout, and finally
        # (3) walking up from this source file. Per-agent kestrel.toml
        # files live alongside agent storage, so checking that directory
        # first is what most deployments expect.
        candidates: list[Path] = []
        storage_path = getattr(self.agent, "storage_path", None) if self.agent else None
        if storage_path:
            sp = Path(storage_path)
            candidates.extend([sp / "kestrel.toml", sp.parent / "kestrel.toml"])
        here = Path(__file__).resolve()
        candidates.extend(parent / "kestrel.toml" for parent in here.parents)

        for candidate in candidates:
            if candidate.exists():
                try:
                    with open(candidate, "rb") as f:
                        data = tomllib.load(f)
                    return data.get("features", {}).get("computer_use", {}) or {}
                except Exception as exc:  # noqa: BLE001
                    logger.warning("failed to read %s: %s", candidate, exc)
                    return {}
        return {}

    def _granted_capabilities(self) -> frozenset[str]:
        """Read Amendment IX grants. Empty set means deny-everything default."""
        for attr in ("granted_capabilities", "capability_grants", "amendment_ix"):
            grants = getattr(self.agent, attr, None)
            if grants:
                return frozenset(grants)

        # The per-agent overlay (self.agent.constitution_text) may grant DANGEROUS
        # capabilities via Amendment IX — but ONLY honor it when the overlay has
        # been integrity-verified against its anchor (#1722). An unanchored or
        # tampered overlay (e.g. one an attacker wrote next to the agent DB to
        # self-grant shell) is ignored; we fall through to the packaged
        # constitution. ``constitution_overlay_verified`` defaults False until
        # verify_constitution_overlay() runs in initialize()/audit.
        # ``is not None`` (not truthiness): an intentionally EMPTY overlay file
        # is ``""`` and must still count as "overlay present" so a verified empty
        # overlay denies all (rather than falling through to package grants) —
        # #1722 codex r4. ``None`` means genuinely no overlay file.
        overlay_text = getattr(self.agent, "constitution_text", None)
        if overlay_text is not None:
            if getattr(self.agent, "constitution_overlay_verified", False):
                # A verified per-agent overlay is AUTHORITATIVE: its parsed grants
                # stand even when EMPTY, so it can NARROW capabilities the
                # packaged constitution would otherwise grant. We do NOT fall
                # through to the package for a verified overlay (#1722 codex r2).
                return parse_amendment_ix_grants(overlay_text)
            logger.warning(
                "Ignoring Amendment IX grants from an UNVERIFIED constitution "
                "overlay (not anchored). Run `kestrel constitution anchor-overlay`."
            )

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
        try:
            from kestrel_sovereign.config import CONSTITUTION_PATH

            return Path(CONSTITUTION_PATH).read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not read constitution: %s", exc)
            return None

    # =========================================================================
    # The five gate steps
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
        self, action: str, payload: Dict[str, Any]
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

    async def _run_gates(
        self,
        *,
        tool_name: str,
        capability: str,
        base_payload: Dict[str, Any],
        path_arg: Optional[str] = None,
        write: bool = False,
        argv: Optional[list[str]] = None,
        pre_approval: Optional[_PreApprovalHook] = None,
    ) -> _GateOutcome:
        """Run the full gate sequence and audit on every refusal.

        ``base_payload`` is the audit payload before path/policy resolution;
        any successful step augments it. The outcome's ``allowed_by`` is
        the chain of gates that passed; on denial the chain ends with
        ``denied:<gate>`` and the audit row is written before returning.

        ``pre_approval`` runs after path-safety + policy have authorized
        the path but **before** the approval queue is asked. It receives
        the resolved realpath and the audit payload so far, and returns
        an augmented payload (e.g. with a diff preview computed from the
        file's current contents). Tools that need to touch the file
        before approval — diff previews, occurrence-finding for fs_edit
        — must do that work here, never in the tool body, so that
        constitutionally-unauthorized callers never trigger any I/O.
        Refusals inside the hook are signalled by raising
        :class:`_PreApprovalRefusal`; the audit row records the reason.
        """
        allowed_by: List[str] = []
        payload = dict(base_payload)

        # 0. Readiness
        if not self._cfg.get("enabled", False):
            await self._audit_denied(tool_name, payload, ["denied:readiness:disabled"])
            return _GateOutcome(False, allowed_by, "readiness:feature not enabled")
        if self._backend is None:
            await self._audit_denied(tool_name, payload, ["denied:readiness:backend"])
            return _GateOutcome(False, allowed_by, "readiness:backend not initialized")

        # 1. Privacy
        if not self._privacy_allows():
            await self._audit_denied(tool_name, payload, ["denied:privacy"])
            return _GateOutcome(False, allowed_by, "privacy:computer_access flag is False")
        allowed_by.append("privacy")

        # 2. Constitution
        if not self._constitution_allows(capability):
            await self._audit_denied(tool_name, payload, allowed_by + ["denied:constitution"])
            return _GateOutcome(
                False,
                allowed_by,
                f"constitution:Amendment IX missing grant '{capability}'",
            )
        allowed_by.append("constitution")

        # 3. Path safety + 4. Policy (filesystem ops)
        if path_arg is not None:
            try:
                resolved = resolve_realpath(path_arg)
            except PathSafetyError as exc:
                await self._audit_denied(
                    tool_name,
                    {**payload, "raw_path": path_arg},
                    allowed_by + ["denied:path_safety"],
                    error=str(exc),
                )
                return _GateOutcome(False, allowed_by, f"path_safety:{exc}")
            payload["path"] = str(resolved)
            allowed_by.append("path_safety")

            decision = self._path_policy.evaluate(resolved, write=write)
            payload["rule"] = decision.rule
            if decision.decision is Decision.DENY:
                await self._audit_denied(tool_name, payload, allowed_by + ["denied:policy"])
                return _GateOutcome(False, allowed_by, f"policy:{decision.rule}")
            allowed_by.append("policy")
            require_approval = decision.decision is Decision.REQUIRE_APPROVAL or write

        # 3. Binary policy (shell)
        elif argv is not None:
            decision = self._binary_policy.evaluate(argv)
            payload["rule"] = decision.rule
            if decision.decision is Decision.DENY:
                await self._audit_denied(tool_name, payload, allowed_by + ["denied:policy"])
                return _GateOutcome(False, allowed_by, f"policy:{decision.rule}")
            allowed_by.append("policy")
            # #1694: honor BinaryPolicy's three-state result instead of
            # hardcoding require_approval=True. Allow-listed binaries
            # (Decision.ALLOW) are pre-approved and bypass the queue —
            # the same contract PathPolicy uses for allow-listed reads
            # under auto_approve_read. Anything not on the allow-list
            # returns REQUIRE_APPROVAL and routes through the queue.
            # #1694 codex review P1: the compound-command guard now
            # lives inside ``BinaryPolicy.evaluate`` itself, so when
            # ``shell`` passes the raw command string the policy
            # already downgrades ALLOW → REQUIRE_APPROVAL on unquoted
            # shell control chars and the rule reads
            # ``compound_command:allow:<bin>``.
            require_approval = decision.decision is Decision.REQUIRE_APPROVAL
        else:
            require_approval = False

        # 4.5. Pre-approval hook (input-dependent work that must run
        # AFTER privacy/constitution/path-safety/policy authorize the
        # path but BEFORE the human approver sees the request).
        if pre_approval is not None and "path" in payload:
            try:
                payload = await pre_approval(Path(payload["path"]), payload)
            except _PreApprovalRefusal as exc:
                chain = allowed_by + [f"denied:input_validation:{exc.reason}"]
                await self._audit_denied(tool_name, payload, chain, error=str(exc))
                return _GateOutcome(
                    False, allowed_by, f"input_validation:{exc.reason}:{exc}"
                )
            allowed_by.append("input_validation")

        # 5. Approval
        if require_approval:
            approved, scope = await self._request_approval(tool_name, payload)
            if not approved:
                await self._audit_denied(
                    tool_name, payload, allowed_by + [f"denied:approval:{scope}"]
                )
                return _GateOutcome(False, allowed_by, f"approval:{scope}")
            allowed_by.append(f"approval:{scope}")

        # Stash the resolved values on the outcome via the payload so
        # callers don't have to re-resolve.
        outcome = _GateOutcome(True, allowed_by, "")
        outcome.payload = payload  # type: ignore[attr-defined]
        return outcome

    async def _audit_denied(
        self,
        tool_name: str,
        payload: Dict[str, Any],
        chain: List[str],
        *,
        error: Optional[str] = None,
    ) -> None:
        if self._audit is None:
            return
        await self._audit.write(
            AuditRecord(
                tool=tool_name,
                backend=self._backend.name if self._backend else "uninitialized",
                args=payload,
                allowed_by=list(chain),
                outcome="denied",
                error=error,
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
        await self._finalize_auto_approve_audit(allowed_by, payload, outcome)

    async def _finalize_auto_approve_audit(
        self,
        allowed_by: List[str],
        payload: Dict[str, Any],
        outcome: str,
    ) -> None:
        """Stamp the real exit code on the auto-approve audit row.

        When a run was auto-approved, ``request_approval`` returned
        ``auto_approve:<id>`` and that id rode the ``approval:<scope>``
        token in ``allowed_by``. Closing the loop here means the audit row
        is the full record the constitution requires — command, agent DID,
        timestamp *and* exit code — for every auto-approved invocation.
        """
        token = next(
            (
                a.split("approval:auto_approve:", 1)[1]
                for a in allowed_by
                if a.startswith("approval:auto_approve:")
            ),
            None,
        )
        if token is None:
            return
        try:
            audit_id = int(token)
        except ValueError:
            return
        rc = payload.get("returncode")
        exit_code = (
            int(rc) if isinstance(rc, int)
            else (0 if outcome == "ok" else 1)
        )
        security = self._get_security_feature()
        store = getattr(security, "permission_store", None) if security else None
        if store is None:
            return
        try:
            await store.finalize_auto_approve(audit_id, exit_code)
        except Exception as exc:  # noqa: BLE001 - audit must not crash a run
            logger.warning(
                "computer_use: failed to finalize auto-approve audit %s: %s",
                audit_id, exc, exc_info=True,
            )

    # =========================================================================
    # Tools
    # =========================================================================

    @tool(
        name="fs_read",
        description="Read a file (allow-list auto-approves; outside list requires human approval).",
        category=ToolCategory.FILE_OPERATIONS,
        command_prefix="!fs-read",
    )
    async def fs_read(self, path: str) -> ToolResult:
        """Read a file the sovereign has authorized.

        Args:
            path: Absolute path or path relative to the current directory.
        """
        outcome = await self._run_gates(
            tool_name="fs-read",
            capability="filesystem_read",
            base_payload={"raw_path": path},
            path_arg=path,
            write=False,
        )
        if not outcome:
            return ToolResult.failed(error=outcome.denied_reason)

        payload = outcome.payload  # type: ignore[attr-defined]
        resolved = Path(payload["path"])
        started = time.monotonic()
        try:
            data = await self._backend.read(resolved, max_bytes=self._max_read_bytes)
            duration_ms = int((time.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="fs-read",
                payload={**payload, "bytes": len(data)},
                allowed_by=outcome.allowed_by,
                outcome="ok",
                duration_ms=duration_ms,
            )
            content_str = data.decode("utf-8", errors="replace")
            # Render the file contents inside the confirmation so the
            # !fs-read CLI surface still shows what the user came for —
            # the command-handler envelope formatter suppresses scalar
            # ``data`` (no list/nested-dict values triggers the
            # structural-payload heuristic), so anything the user is
            # expected to see has to live in ``confirmation``.
            confirmation = (
                f"Read {len(data)} bytes from {resolved}:\n"
                + (content_str if content_str else "(empty file)")
            )
            return ToolResult.ok(
                confirmation,
                data={
                    "path": str(resolved),
                    "bytes": len(data),
                    "content": content_str,
                },
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="fs-read",
                payload=payload,
                allowed_by=outcome.allowed_by,
                outcome="error",
                duration_ms=duration_ms,
                error=str(exc),
            )
            return ToolResult.failed(error=str(exc))

    @tool(
        name="fs_list",
        description="List a directory (allow-list auto-approves; outside list requires human approval).",
        category=ToolCategory.FILE_OPERATIONS,
        command_prefix="!fs-list",
    )
    async def fs_list(self, path: str) -> ToolResult:
        """List a directory the sovereign has authorized.

        Args:
            path: Absolute path or path relative to the current directory.
        """
        outcome = await self._run_gates(
            tool_name="fs-list",
            capability="filesystem_read",
            base_payload={"raw_path": path},
            path_arg=path,
            write=False,
        )
        if not outcome:
            return ToolResult.failed(error=outcome.denied_reason)

        payload = outcome.payload  # type: ignore[attr-defined]
        resolved = Path(payload["path"])
        started = time.monotonic()
        try:
            entries = await self._backend.list(resolved)
            duration_ms = int((time.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="fs-list",
                payload={**payload, "count": len(entries)},
                allowed_by=outcome.allowed_by,
                outcome="ok",
                duration_ms=duration_ms,
            )
            return ToolResult.ok(
                f"Listed {len(entries)} entry(ies) under {resolved}.",
                data={
                    "path": str(resolved),
                    "entries": [e.__dict__ for e in entries],
                },
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="fs-list",
                payload=payload,
                allowed_by=outcome.allowed_by,
                outcome="error",
                duration_ms=duration_ms,
                error=str(exc),
            )
            return ToolResult.failed(error=str(exc))

    @tool(
        name="fs_write",
        description="Replace the contents of a file (always approval-gated).",
        category=ToolCategory.FILE_OPERATIONS,
        command_prefix="!fs-write",
    )
    async def fs_write(self, path: str, content: str) -> ToolResult:
        """Write to a file (always approval-gated).

        The diff preview shown to the human approver is computed inside
        the pre-approval hook so the existing file is **only read after**
        privacy/constitution/path-safety/policy have authorized the path.

        Args:
            path: Absolute path or path relative to the current directory.
            content: UTF-8 content that will replace the file's body.
        """
        new_bytes = content.encode("utf-8")

        async def _prepare(resolved: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
            return {
                **payload,
                "diff_preview": _diff_preview(resolved, new_bytes),
            }

        outcome = await self._run_gates(
            tool_name="fs-write",
            capability="filesystem_write",
            base_payload={"raw_path": path, "bytes": len(new_bytes)},
            path_arg=path,
            write=True,
            pre_approval=_prepare,
        )
        if not outcome:
            return ToolResult.failed(error=outcome.denied_reason)

        payload = outcome.payload  # type: ignore[attr-defined]
        resolved = Path(payload["path"])
        started = time.monotonic()
        try:
            written = await self._backend.write(resolved, new_bytes)
            duration_ms = int((time.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="fs-write",
                payload={"path": str(resolved), "bytes": written, "rule": payload.get("rule")},
                allowed_by=outcome.allowed_by,
                outcome="ok",
                duration_ms=duration_ms,
            )
            return ToolResult.ok(
                f"Wrote {written} bytes to {resolved}.",
                data={"path": str(resolved), "bytes_written": written},
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="fs-write",
                payload={"path": str(resolved), "rule": payload.get("rule")},
                allowed_by=outcome.allowed_by,
                outcome="error",
                duration_ms=duration_ms,
                error=str(exc),
            )
            return ToolResult.failed(error=str(exc))

    @tool(
        name="fs_edit",
        description="Replace one occurrence of old_text with new_text in a file (always approval-gated).",
        category=ToolCategory.FILE_OPERATIONS,
        command_prefix="!fs-edit",
    )
    async def fs_edit(
        self, path: str, old_text: str, new_text: str, occurrence: int = 1
    ) -> ToolResult:
        """Targeted in-place edit of a file.

        The tool replaces the ``occurrence``-th instance of ``old_text``
        with ``new_text``. **All file I/O happens inside a pre-approval
        hook that runs only after privacy/constitution/path-safety/policy
        have authorized the path.** Constitutionally-unauthorized callers
        cannot use this tool to probe file existence, readability, UTF-8
        validity, or substring presence.

        Args:
            path: File to edit.
            old_text: Exact text to find (must match exactly).
            new_text: Replacement text.
            occurrence: Which occurrence to replace, 1-indexed (default 1).
        """
        # We carry the computed write payload across the gate boundary
        # via this dict — it's populated inside ``_prepare`` once gates
        # have authorized the file read.
        prepared: Dict[str, Any] = {}

        async def _prepare(resolved: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
            # Parameter validation runs here too: an unauthorized caller
            # mustn't be able to learn that occurrence=0 is rejected.
            if occurrence < 1:
                raise _PreApprovalRefusal(
                    "occurrence", "occurrence must be >= 1"
                )
            try:
                old_bytes = resolved.read_bytes() if resolved.exists() else b""
            except OSError as exc:
                raise _PreApprovalRefusal("read", f"read failed: {exc}")
            try:
                old_str = old_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raise _PreApprovalRefusal(
                    "encoding", "fs_edit only supports UTF-8 files"
                )

            idx = -1
            cursor = 0
            for _ in range(occurrence):
                idx = old_str.find(old_text, cursor)
                if idx < 0:
                    raise _PreApprovalRefusal(
                        "missing_text",
                        f"old_text not found (occurrence {occurrence})",
                    )
                cursor = idx + len(old_text)
            new_str = old_str[:idx] + new_text + old_str[idx + len(old_text):]
            new_bytes = new_str.encode("utf-8")

            prepared["new_bytes"] = new_bytes
            return {
                **payload,
                "old_bytes": len(old_bytes),
                "new_bytes": len(new_bytes),
                "diff_preview": _diff_preview_from_strings(old_str, new_str, str(resolved)),
            }

        outcome = await self._run_gates(
            tool_name="fs-edit",
            capability="filesystem_write",
            base_payload={"raw_path": path, "occurrence": occurrence},
            path_arg=path,
            write=True,
            pre_approval=_prepare,
        )
        if not outcome:
            return ToolResult.failed(error=outcome.denied_reason)

        payload = outcome.payload  # type: ignore[attr-defined]
        resolved = Path(payload["path"])
        new_bytes = prepared["new_bytes"]
        started = time.monotonic()
        try:
            written = await self._backend.write(resolved, new_bytes)
            duration_ms = int((time.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="fs-edit",
                payload={
                    "path": str(resolved),
                    "bytes": written,
                    "rule": payload.get("rule"),
                    "occurrence": occurrence,
                },
                allowed_by=outcome.allowed_by,
                outcome="ok",
                duration_ms=duration_ms,
            )
            return ToolResult.ok(
                f"Edited {resolved} (occurrence {occurrence}, {written} bytes written).",
                data={
                    "path": str(resolved),
                    "bytes_written": written,
                    "occurrence": occurrence,
                },
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="fs-edit",
                payload={"path": str(resolved), "rule": payload.get("rule")},
                allowed_by=outcome.allowed_by,
                outcome="error",
                duration_ms=duration_ms,
                error=str(exc),
            )
            return ToolResult.failed(error=str(exc))

    @tool(
        name="shell",
        description=(
            "Run a shell command. Deny-listed binaries hard-refuse; "
            "auto-approved binaries run without a prompt; everything "
            "else routes through the ApprovalQueue."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!shell",
    )
    async def shell(self, command: str, timeout: int | str = 60) -> ToolResult:
        """Run a shell command after policy + (conditional) approval.

        Approval semantics (#1694):

        - Deny-listed binary → hard refuse before queueing.
        - Auto-approved binary (allow-list match) → runs without a
          prompt. Same contract as ``auto_approve_read`` for paths.
        - Anything else → routes through ApprovalQueue (operator or
          scoped auto-approve rule decides).

        The compound-command guard (raw string with unquoted ``;``,
        ``&&``, backticks, ``$(...)``, redirects, newline) downgrades
        ALLOW to REQUIRE_APPROVAL so an allow-listed first token can't
        bless a piggy-backed second command.

        Args:
            command: The shell command to run; tokenized with shlex.
            timeout: Wall-clock seconds before the process is killed. Coerced
                to int at the boundary; a non-numeric value is rejected.

        Returns:
            ToolResult.ok when the command exits 0; PARTIAL when the
            command ran but exited non-zero or timed out (the LLM
            should NOT claim success — but the shell did run, which
            matters for audit and for follow-up steps that read
            stdout/stderr); ERROR for empty-command, gate denial,
            or any backend exception.
        """
        argv = split_command(command)
        if not argv:
            return ToolResult.failed(error="empty command")

        # The LLM may pass ``timeout`` as a string (e.g. "60"); the backend
        # does numeric comparisons on it (asyncio.wait_for's ``<= 0`` check),
        # so coerce to int at the boundary and reject non-numeric / non-positive
        # values rather than letting a TypeError surface deep in the exec path.
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            return ToolResult.failed(
                error=f"timeout must be an integer number of seconds, got {timeout!r}"
            )
        if timeout <= 0:
            return ToolResult.failed(
                error=f"timeout must be a positive number of seconds, got {timeout}"
            )

        # Capability depends on which backend is wired up; gate semantics
        # treat the two as distinct constitutional grants.
        capability = (
            "shell_execution_host"
            if self._backend is not None and self._backend.name == "local"
            else "shell_execution_sandboxed"
        )

        outcome = await self._run_gates(
            tool_name="shell",
            capability=capability,
            base_payload={
                "argv": argv,
                "binary": Path(argv[0]).name,
                "backend": self._backend.name if self._backend else "uninitialized",
                "timeout": timeout,
            },
            # Pass the RAW command (not the pre-tokenized argv) so
            # ``BinaryPolicy.evaluate`` can apply its compound-command
            # guard (#1694 codex review P1). Argv-list inputs are
            # trusted; raw strings get the metacharacter check.
            argv=command,
        )
        if not outcome:
            return ToolResult.failed(error=outcome.denied_reason)

        payload = outcome.payload  # type: ignore[attr-defined]
        started = time.monotonic()
        try:
            result = await self._backend.exec(argv, cwd=None, env=None, timeout=timeout)
            duration_ms = int((time.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="shell",
                payload={**payload, "returncode": result.returncode},
                allowed_by=outcome.allowed_by,
                outcome="ok" if result.returncode == 0 else "error",
                duration_ms=duration_ms,
                error=None if result.returncode == 0 else f"exit {result.returncode}",
            )
            data = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_ms": result.duration_ms,
                "timed_out": result.timed_out,
            }
            # Render stdout/stderr inside the confirmation so the
            # !shell CLI surface keeps showing the command output —
            # the command-handler envelope formatter suppresses
            # scalar-only ``data`` (the structural-payload heuristic
            # only fires on list/nested-dict values), so the user-
            # visible payload has to live in ``confirmation``.
            stdout_block = (
                f"\nstdout:\n{result.stdout}" if result.stdout else ""
            )
            stderr_block = (
                f"\nstderr:\n{result.stderr}" if result.stderr else ""
            )
            if result.returncode == 0:
                return ToolResult.ok(
                    f"Command ran successfully (rc=0, {result.duration_ms}ms)."
                    + stdout_block
                    + stderr_block,
                    data=data,
                )
            # Non-zero exit: PARTIAL — the shell ran (so audit / follow-up
            # tooling that reads stdout/stderr is meaningful) but the
            # LLM must not claim success.
            stderr_tail = (result.stderr or "")[-200:].strip()
            caveat = (
                f"command exited rc={result.returncode}"
                + (" (timed out)" if result.timed_out else "")
                + (f"; stderr tail: {stderr_tail}" if stderr_tail else "")
            )
            return ToolResult.partial(
                f"Command ran but failed (rc={result.returncode}, {result.duration_ms}ms)."
                + stdout_block
                + stderr_block,
                caveat,
                data=data,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.monotonic() - started) * 1000)
            await self._audit_run(
                tool_name="shell",
                payload=payload,
                allowed_by=outcome.allowed_by,
                outcome="error",
                duration_ms=duration_ms,
                error=str(exc),
            )
            return ToolResult.failed(error=str(exc))


def _diff_preview(path: Path, new_bytes: bytes, *, max_chars: int = 4000) -> str:
    """Best-effort textual preview of what a write will change.

    NOTE: this reads ``path`` from disk to compute the diff. Callers that
    are pre-gate (e.g. ``fs_write``'s base_payload construction) MUST be
    okay with that read happening before privacy/constitution refuse the
    call — for ``fs_write`` we accept the trade because the agent already
    supplied the path. ``fs_edit`` uses :func:`_diff_preview_from_strings`
    instead so its read happens inside the pre-approval hook (after the
    gates have authorized the path).
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

    return _diff_preview_from_strings(old_text, new_text, str(path), max_chars=max_chars)


def _diff_preview_from_strings(
    old_text: str, new_text: str, path: str, *, max_chars: int = 4000
) -> str:
    """Render a unified diff from two already-read strings. No I/O."""
    import difflib

    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=path + "@old",
        tofile=path + "@new",
        n=2,
    )
    rendered = "".join(diff)
    if len(rendered) > max_chars:
        return rendered[:max_chars] + f"\n... [truncated {len(rendered) - max_chars} chars]"
    return rendered or "(no textual diff)"
