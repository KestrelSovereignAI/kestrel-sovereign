"""TalonCoordinatorFeature - lightweight dispatch to external Talon daemon.

Wraps coordination, not the engine. Dispatches work to the external
kestrel-talon daemon via Agent Mesh Protocol (preferred) or CLI fallback.

Reference: sovereign #301
"""

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.waits import run_wait_loop
from kestrel_sovereign.waits.reconciler import register_wait_watch
from kestrel_sovereign.features.talon.wait_provider import TalonWaitable
# Mesh is gone (#1367 phase 5). Talon dispatch now uses the A2A
# task-submission path — same wire endpoint as send_a2a_task on
# PeersFeature, but called directly because coordinator dispatch
# happens at the feature level (not from an LLM tool turn).
from kestrel_sovereign.features.talon.runtime import (
    TalonExecution,
    TalonPolicy,
    TalonPreference,
    TalonRuntimeError,
    TalonRuntimeRequest,
    build_talon_invocation,
    load_talon_policy_preference,
    normalize_auth_lane,
    normalize_backend,
    parse_talon_bool,
    resolve_runtime,
    sanitize_env_for_backend,
    write_talon_preference,
)
from kestrel_sovereign.features.talon.verification import (
    CommandExecution,
    TalonVerifier,
    TestCommandResult,
    VerificationEvidence,
    VerificationState,
)

logger = logging.getLogger(__name__)


# Default sibling-checkout layout assumed throughout: kestrel-sovereign
# and target repos live as siblings under a common project parent.
_DEFAULT_PROJECT_PARENT = Path(__file__).resolve().parents[4]

# The running agent's own source root. Captured at import time so an
# in-process attacker mutating ``Path(__file__)`` later can't move it.
# The dispatcher refuses to point Talon at this directory; talon
# operates against a separate workspace clone instead. See
# ``_workspace_root_for`` and ``talon_setup_workspace``.
_RUNNING_AGENT_SOURCE_ROOT = Path(__file__).resolve().parents[3]

# Talon's reserved issue-lifecycle labels. kestrel-talon uses
# ``agent-claimed`` as its "this issue is claimed" marker:
# ``GitHubClient.is_claimed()`` returns True iff that label is present,
# and ``kestrel-talon claim`` aborts with "Issue #N is already claimed"
# before doing any work. So a file-then-claim primitive must NOT stamp
# any of these at creation time — Talon applies ``agent-claimed`` itself
# when it claims.
#
# This is the COMPLETE set, pinned as an exact 1:1 mirror of every
# ``label_*`` default in kestrel-talon ``kestreltalon/config.py``:
#
#     label_analyzing  = "agent-analyzing"
#     label_clarifying = "agent-clarifying"
#     label_in_progress = "agent-claimed"   # the is_claimed() marker
#     label_blocked   = "agent-blocked"
#     label_failed    = "agent-failed"
#     label_completed = "agent-complete"
#
# (Note: ``label_in_progress`` resolves to ``agent-claimed`` — there is
# no separate ``agent-in-progress`` label; the claimed marker IS the
# in-progress marker.) Pinned, not imported: ``kestreltalon`` is invoked
# as a CLI binary and is not an importable module in this venv, and
# sibling-checkout path coupling is forbidden (no local-path deps). If
# Talon ever adds/renames a lifecycle label, update this set to match.
_TALON_RESERVED_LABELS = frozenset({
    "agent-analyzing",
    "agent-clarifying",
    "agent-claimed",   # == kestrel-talon label_in_progress
    "agent-blocked",
    "agent-failed",
    "agent-complete",
})


# Discriminated reason codes for ``talon_file_and_claim`` failures.
# Each one names a distinct fix path the agent (or the operator) can act
# on. They're public values, not just internal hints — every failed
# ToolResult carries one in ``data['reason_code']`` AND in the top-level
# error string, so the model's narration can surface "MISSING_GH_AUTH —
# please authenticate" instead of the historical catch-all "may have
# been denied at the approval gate, gh is not authenticated, or it
# failed for a non-label reason" (#1383).
TALON_FAC_REASONS = {
    "MISSING_COMPUTER_USE",  # ComputerUseFeature not enabled
    "GATE_DENIED",           # approval gate refused the shell
    "MISSING_GH_AUTH",       # `gh` not authenticated / no token
    "GH_NOT_INSTALLED",      # the `gh` binary itself is absent
    "REPO_NOT_FOUND",        # repo not visible to the auth'd user
    "LABEL_REJECTED",        # label unknown/invalid even after retry
    "SHELL_TIMEOUT",         # shell call hit the 120s wall
    "URL_PARSE_FAILED",      # gh exited 0 but URL regex missed
    "DISPATCH_FAILED",       # talon_claim never reported dispatched
    "UNKNOWN_FAILURE",       # fallthrough
}


def _gh_failure_reason(
    shell_res,
    stdout: str,
    stderr: str,
    *,
    succeeded: bool,
    parsed_url: bool,
) -> tuple[str, str]:
    """Classify a ``gh issue create`` failure into a stable reason code.

    The shell call goes through ``ComputerUseFeature.shell``, which
    returns:
      * ``ok``      — rc=0  (stdout/stderr on ``data``)
      * ``partial`` — ran but rc!=0 (caveat in ``error``; ``data`` has rc/timeout)
      * ``failed``  — gate denied / backend exception / empty argv

    We sniff stderr/stdout/error for the patterns the actual ``gh`` CLI
    emits (and the gate's ``denied_reason``) and pick the most specific
    code. The returned ``(code, hint)`` tuple is rendered into the
    ``ToolResult.failed`` error string so the agent's narration carries
    something actionable.
    """
    error_blob = (shell_res.error or "") if shell_res else ""
    full = f"{stderr}\n{stdout}\n{error_blob}".lower()
    timed_out = bool(
        getattr(shell_res, "data", None)
        and shell_res.data.get("timed_out")
    )

    if timed_out:
        return (
            "SHELL_TIMEOUT",
            "`gh issue create` exceeded the 120s wall. Retry, or check "
            "network reachability to api.github.com.",
        )

    # Stderr/stdout patterns are checked BEFORE shell-status branches so
    # that a non-zero exit (partial) with "Bad credentials" in stderr is
    # classified as MISSING_GH_AUTH instead of falling through to a
    # generic UNKNOWN_FAILURE. The patterns are ordered most-specific
    # first.
    if "gh: command not found" in full or (
        "no such file" in full and "gh" in full
    ):
        return (
            "GH_NOT_INSTALLED",
            "the `gh` CLI is not on PATH for this agent. Install gh in "
            "the host environment.",
        )

    if (
        "bad credentials" in full
        or "gh auth login" in full
        or "no github token" in full
        or "authentication required" in full
        or "http 401" in full
        or " 401" in full
        or "401 unauthorized" in full
    ):
        return (
            "MISSING_GH_AUTH",
            "`gh` is not authenticated. Set GH_TOKEN / GITHUB_TOKEN, or "
            "run `gh auth login` in the host environment.",
        )

    if (
        "could not resolve to a repository" in full
        or ("not found" in full and "repository" in full)
        or "http 404" in full
        or "404 not found" in full
    ):
        return (
            "REPO_NOT_FOUND",
            "GitHub returned 404 / not-found for the repo. Verify the "
            "owner/name and that the auth'd user has access.",
        )

    if "label" in full and (
        "not found" in full
        or "could not add" in full
        or "not a valid" in full
        or "no label" in full
    ):
        return (
            "LABEL_REJECTED",
            "gh rejected one of the labels even after the no-label retry. "
            "Check that the repo accepts the labels you passed.",
        )

    if shell_res is not None and str(shell_res.status) in (
        "error", "failed", "ToolResultStatus.ERROR",
    ):
        # Backend exception, empty argv, or gate denial — surface via
        # ``shell_res.error`` when ``data`` was empty. (ToolResult.failed
        # maps to status "error"; we accept "failed" too so test fakes
        # that pass the historical string still classify correctly.)
        elow = error_blob.lower()
        if "denied" in elow or "approval" in elow or "policy" in elow:
            return (
                "GATE_DENIED",
                "the approval-gate denied the shell. Allowlist the "
                "`gh issue create` pattern or escalate to the operator.",
            )
        if "empty command" in elow:
            return (
                "GH_NOT_INSTALLED",
                "shell argv was empty — likely shlex parse failure.",
            )

    if succeeded and not parsed_url:
        return (
            "URL_PARSE_FAILED",
            "`gh issue create` exited 0 but stdout did not contain a "
            "parseable issue URL. The repo may use an unusual gh config.",
        )

    return (
        "UNKNOWN_FAILURE",
        "shell ran but did not produce a parseable issue URL. Inspect "
        "stderr_tail / stdout_tail / shell_error in the structured data.",
    )


def _path_contains(parent: Path, child: Path) -> bool:
    """True iff ``child`` is ``parent`` or under it. Resolves both."""
    try:
        parent_r = parent.resolve()
        child_r = child.resolve()
    except OSError:
        return False
    if parent_r == child_r:
        return True
    return parent_r in child_r.parents


class TalonCoordinatorFeature(Feature):
    """Thin dispatcher to the external kestrel-talon daemon.

    Provides !talon commands for claiming issues, batch processing,
    and checking status. Prefers mesh dispatch over CLI fallback.
    """

    # Conservative ceiling on a single blocking ``talon_wait``. Waits
    # longer than this should rely on the generic wait reconciler cron
    # signal to wake the agent rather than holding a turn open. The
    # reconciler drives the talon.job_complete wake via TalonWaitable
    # (Wave 2 of #1860) — the talon-specific talon_monitor cron is retired.
    _TALON_WAIT_MAX_SECONDS = 3600

    def __init__(self, agent):
        super().__init__(agent)
        # message_id -> {pid, started_at, log_path, command, repo,
        # issue, status, returncode, completed_at, process}
        self._jobs: Dict[str, Dict[str, Any]] = {}
        # Eager reload so a fresh feature instance immediately sees
        # jobs from a previous process — dispatch-then-persist would
        # otherwise truncate the registry to the new job alone.
        self._reload_persisted_jobs()

    @property
    def tool_description(self) -> str:
        return "Dispatch work to Talon autonomous coding agent and monitor status"

    async def initialize(self):
        logger.info("TalonCoordinatorFeature initialized")
        # Self-register the talon.job_complete signal source on the
        # agent's signal registry. Owning the registration here (vs.
        # in agent boot) keeps it scoped to this feature so when
        # Talon eventually extracts to an external feature package
        # the registration travels with it. No-op if signal_registry
        # is absent (test stubs, headless agents) — and idempotent
        # on a second initialize() since a duplicate registration
        # would otherwise raise and shadow real-failure warnings.
        registry = getattr(self.agent, "signal_registry", None)
        if registry is not None and hasattr(registry, "register"):
            from kestrel_sovereign.signals.sources.talon import (
                SOURCE_NAME as _TALON_SOURCE_NAME,
                build_talon_job_complete_registration,
            )
            already = (
                hasattr(registry, "get")
                and registry.get(_TALON_SOURCE_NAME) is not None
            )
            if not already:
                try:
                    registry.register(
                        build_talon_job_complete_registration()
                    )
                except Exception as e:
                    logger.warning(
                        "TalonCoordinatorFeature could not register "
                        "talon.job_complete signal source: %s", e,
                    )

    async def post_all_features_loaded(self, agent):
        """Register the ``talon:`` Waitable provider with the wait engine.

        Lets ``wait("talon:<job_id>")`` dispatch here, and lets the
        Wave-2 reconciler enumerate this kind. ``talon_wait`` calls the
        engine directly and does not depend on this registration.
        """
        registry = getattr(agent, "wait_registry", None)
        if registry is not None:
            registry.register(TalonWaitable(self), replace=True)

        # One-time migration of legacy talon_monitor dedup state into the
        # generic reconciler ledger. Pre-Wave-2 jobs.json rows carry
        # ``last_signaled_status`` from the retired monitor; without seeding
        # the (initially empty) ledger, the first wait_reconcile tick would
        # re-fire talon.job_complete for every already-delivered terminal job
        # (codex Wave 2 P2). seed_signaled is INSERT-OR-IGNORE, so this is a
        # safe no-op on every subsequent startup.
        await self._seed_legacy_signal_ledger(agent)

    async def _seed_legacy_signal_ledger(self, agent) -> None:
        """Seed wait_signal_state from legacy jobs.json ``last_signaled_status``."""
        raw_storage = getattr(agent, "_raw_storage", None)
        db = getattr(raw_storage, "db", None)
        if db is None:
            return
        try:
            from kestrel_sdk.tools import Outcome
            from kestrel_sovereign.storage.async_wait_signal_store import (
                WaitSignalStore,
            )

            agent_id = (
                getattr(agent, "did", None)
                or getattr(agent, "agent_id", None)
                or ""
            )
            store = WaitSignalStore(db, str(agent_id))
            self._reload_persisted_jobs()
            for job_id, info in self._jobs.items():
                legacy = info.get("last_signaled_status")
                if not legacy:
                    continue
                # Map the legacy talon status onto the generic Outcome the
                # reconciler dedups on (complete -> done; the rest -> failed).
                outcome = (
                    Outcome.DONE if legacy == "complete" else Outcome.FAILED
                )
                await store.seed_signaled("talon", str(job_id), outcome.value)
        except Exception as e:  # never let a migration hiccup block startup
            logger.warning(
                "TalonCoordinatorFeature: legacy signal-ledger seed failed: %s",
                e,
            )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool(
        name="talon_claim",
        description=(
            "Dispatch a single issue to Talon for autonomous "
            "implementation. Returns immediately with a job_id; the "
            "actual work runs in the background. Poll talon_status "
            "or talon_job_log to follow progress."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!talon claim",
    )
    async def talon_claim(
        self,
        repo: str,
        issue: int,
        max_iterations: Optional[int] = None,
        max_turns: Optional[int] = None,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        auth_lane: Optional[str] = None,
        skip_clarification: Optional[bool] = None,
        worktree: bool = True,
        self_review: Optional[bool] = None,
    ) -> ToolResult:
        """Claim an issue for Talon to implement.

        Talon's claim flow runs the LLM agent loop, runs quality
        gates, commits, pushes, and opens a PR — typically 10-30
        minutes per issue. This dispatcher launches it in the
        background and returns immediately. The previous synchronous
        implementation waited up to 300s for completion and killed
        Talon mid-implementation; that's why earlier dispatches
        looked like they "did nothing."

        Args:
            repo: GitHub repo in owner/name format
                (e.g. ``KestrelSovereignAI/kestrel-sovereign``). The
                special string ``"self"`` resolves to
                ``KestrelSovereignAI/kestrel-sovereign``.
            issue: Issue number to claim.
            max_iterations: Max LLM implementation iterations. If unset,
                uses ``[talon.preference].max_iterations``.
            max_turns: Max agent turns per Talon iteration. If unset,
                uses ``[talon.preference].max_turns``.
            backend: Talon runtime backend: ``claude``, ``codex``, or
                ``opencode``. This is separate from Kestrel chat LLM routing.
            model: Backend-specific model. Claude accepts ``opus``,
                ``sonnet``, or ``haiku``; Codex accepts current Codex model
                IDs; OpenCode accepts provider/model IDs.
            auth_lane: ``oauth``, ``api_key``, or ``provider_config``.
            skip_clarification: If True, skip the analysis/clarification
                phase. Recommended when nobody is watching to answer
                questions; the issue body should already be specific.
            worktree: If True, run in an isolated git worktree so the
                target checkout isn't clobbered. README marks this
                "strongly recommended" — don't disable unless you
                know what you're doing.

        Returns:
            ``{"dispatched": True, "method": "cli_background",
            "job_id": ..., "log_path": ..., "pid": ...}`` on success.
            Failure returns ``{"dispatched": False, "error": ...}``.
        """
        try:
            policy, preference = load_talon_policy_preference()
            runtime_request = TalonRuntimeRequest(
                backend=normalize_backend(backend),
                model=model,
                auth_lane=normalize_auth_lane(auth_lane),
            )
        except TalonRuntimeError as e:
            return ToolResult.failed(
                str(e),
                data={
                    "dispatched": False,
                    "state": "invalid_talon_runtime",
                    "error": str(e),
                },
            )

        try:
            resolved_backend, resolved_model, resolved_auth_lane = resolve_runtime(
                runtime_request,
                preference,
                policy,
            )
        except TalonRuntimeError as e:
            return ToolResult.failed(
                str(e),
                data={
                    "dispatched": False,
                    "state": "talon_policy_rejected",
                    "error": str(e),
                },
            )

        # A2A dispatch only carries repo/issue today, so using it for a
        # non-default runtime would silently ignore the agent's Talon controls.
        use_a2a = (
            resolved_backend == "claude"
            and resolved_model == "opus"
            and resolved_auth_lane == "oauth"
            and backend is None
            and model is None
            and auth_lane is None
        )
        if use_a2a:
            a2a_result = await self._dispatch_via_a2a(repo, issue)
            if a2a_result.get("dispatched"):
                tracking_id = (
                    a2a_result.get("job_id")
                    or a2a_result.get("task_id")
                    or "?"
                )
                return ToolResult.ok(
                    confirmation=(
                        f"Dispatched {repo}#{issue} to talon via A2A "
                        f"(task_id={tracking_id})"
                    ),
                    data=a2a_result,
                )

        repo_resolved = self._resolve_repo(repo)
        workspace = self._workspace_path_for(repo_resolved)

        unsafe_reason = (
            self._assert_workspace_safe(workspace)
            if policy.require_sandboxed_workspace
            else None
        )
        if unsafe_reason:
            return ToolResult.failed(
                unsafe_reason,
                data={
                    "dispatched": False,
                    "state": "unsafe_workspace",
                    "error": unsafe_reason,
                },
            )

        # If the workspace doesn't exist, refuse and tell the agent
        # the structural next step. Don't silently fall through to
        # the running source tree — that's the bug we're fixing.
        state = self._workspace_state(repo_resolved)
        if not state["exists"] or not state["is_git"]:
            err_msg = (
                "No talon workspace exists for "
                f"{repo_resolved} at {workspace}. The dispatcher "
                "will not operate on the running agent's source "
                "tree. Call talon_setup_workspace(repo) to "
                "provision a sandboxed clone, then retry."
            )
            return ToolResult.failed(
                err_msg,
                data={
                    "dispatched": False,
                    "state": "workspace_not_provisioned",
                    "error": err_msg,
                    "workspace": state,
                    "next_step": (
                        f"talon_setup_workspace(repo='{repo_resolved}')"
                    ),
                },
            )

        worktree_base = (
            os.environ.get("KESTREL_TALON_WORKTREE_BASE")
            or str(workspace.parent)
        )

        try:
            execution = TalonExecution(
                repo=repo_resolved,
                issue=issue,
                repo_dir=workspace,
                worktree_base=Path(worktree_base),
                worktree=worktree,
                max_iterations=(
                    int(max_iterations)
                    if max_iterations is not None
                    else preference.max_iterations
                ),
                max_turns=(
                    int(max_turns)
                    if max_turns is not None
                    else preference.max_turns
                ),
                skip_clarification=(
                    parse_talon_bool(skip_clarification, "skip_clarification")
                    if skip_clarification is not None
                    else preference.skip_clarification
                ),
                self_review=(
                    parse_talon_bool(self_review, "self_review")
                    if self_review is not None
                    else preference.self_review
                ),
            )
            invocation = build_talon_invocation(
                runtime_request,
                execution,
                policy=policy,
                preference=preference,
            )
        except TalonRuntimeError as e:
            return ToolResult.failed(
                str(e),
                data={
                    "dispatched": False,
                    "state": "talon_policy_rejected",
                    "error": str(e),
                },
            )

        cli_result = await self._dispatch_via_cli_background(
            invocation.argv,
            label=f"claim:{repo_resolved}#{issue}",
            env=invocation.env,
            extra_meta={
                "repo": repo_resolved,
                "issue": issue,
                "workspace": str(workspace),
                **invocation.metadata(),
            },
        )

        if cli_result.get("dispatched"):
            return ToolResult.ok(
                confirmation=(
                    f"Dispatched {repo_resolved}#{issue} to talon via CLI "
                    f"background (job_id={cli_result.get('job_id', '?')}, "
                    f"pid={cli_result.get('pid', '?')})"
                ),
                data=cli_result,
            )
        return ToolResult.failed(
            cli_result.get("error") or "talon CLI dispatch failed",
            data=cli_result,
        )

    @tool(
        name="talon_file_and_claim",
        description=(
            "Loop-closing primitive: file a GitHub issue, then "
            "immediately dispatch it to Talon. Runs `gh issue create` "
            "through the audited computer_use shell path (auto-approved "
            "if the Sovereign added a matching allowlist pattern), parses "
            "the new issue number, then calls talon_claim. Returns the "
            "issue URL and the Talon job_id."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!talon file-and-claim",
    )
    async def talon_file_and_claim(
        self,
        title: str,
        body: str,
        labels: Optional[str] = None,
        repo: str = "KestrelSovereignAI/kestrel-sovereign",
    ) -> ToolResult:
        """File an issue and dispatch it to Talon in one hop.

        This is the single primitive that closes Emma's loop: instead of
        the Sovereign typing `gh issue create` and then a separate claim,
        the agent calls this once. The `gh` invocation goes through
        ``ComputerUseFeature.shell`` so the scoped auto-approve policy
        (epic #1290 / D1) governs it and a full audit row is written —
        the agent never shells out unaudited.

        Args:
            title: Issue title.
            body: Issue body (markdown).
            labels: Optional comma-separated label names. Talon's
                reserved lifecycle labels (``agent-claimed`` etc.) are
                dropped automatically — Talon applies ``agent-claimed``
                itself when it claims, and pre-stamping it would make the
                claim abort as "already claimed".
            repo: ``owner/name`` (default the sovereign repo; ``"self"``
                resolves the same way ``talon_claim`` does).

        Returns:
            ``ToolResult.ok`` with ``{filed, issue_url, issue_number,
            job_id, dispatched}``. ``failed`` if the issue could not be
            created or its number could not be parsed; ``partial`` if the
            issue was filed but the Talon dispatch did not take.
        """
        repo_resolved = self._resolve_repo(repo)

        agent = getattr(self, "agent", None)
        cu = (
            agent.get_feature("ComputerUseFeature")
            if agent is not None and hasattr(agent, "get_feature")
            else None
        )
        if cu is None:
            await self._log_fac_outcome(
                decision="missing_computer_use",
                reason_code="MISSING_COMPUTER_USE",
                repo=repo_resolved,
                filed=False,
                dispatched=False,
            )
            return ToolResult.failed(
                "talon_file_and_claim: MISSING_COMPUTER_USE — "
                "ComputerUseFeature unavailable, refusing to file the "
                "issue via an unaudited shell. Enable computer_use so the "
                "scoped auto-approve policy and audit row apply.",
                data={
                    "filed": False,
                    "dispatched": False,
                    "reason_code": "MISSING_COMPUTER_USE",
                    "repo": repo_resolved,
                },
            )

        # Drop Talon's reserved lifecycle labels (esp. ``agent-claimed``).
        # Filing the issue with ``agent-claimed`` makes the very next
        # ``talon_claim`` abort — Talon's ``is_claimed()`` sees the label
        # and reports "Issue #N is already claimed", so the loop never
        # closes (root cause of the #1299/#1301/#1303 Talon failures).
        # Talon stamps ``agent-claimed`` itself as part of claiming.
        requested = [
            l.strip() for l in (labels or "").split(",") if l.strip()
        ]
        applied_labels = [
            l for l in requested
            if l.lower() not in _TALON_RESERVED_LABELS
        ]
        stripped_labels = [
            l for l in requested
            if l.lower() in _TALON_RESERVED_LABELS
        ]

        async def _run_create(cmd_labels: List[str]):
            parts = [
                "gh", "issue", "create",
                "-R", repo_resolved,
                "--title", title,
                "--body", body,
            ]
            for lab in cmd_labels:
                parts += ["--label", lab]
            res = await cu.shell(shlex.join(parts), timeout=120)
            d = res.data or {}
            out = str(d.get("stdout", ""))
            err = str(d.get("stderr", ""))
            ok = res.status == "ok"
            mm = re.search(r"https://github\.com/\S+/issues/(\d+)", out)
            return res, ok, out, err, mm

        shell_res, succeeded, stdout, stderr, m = await _run_create(
            applied_labels
        )

        # A loop-closing primitive must not die on a bad label. ``gh
        # issue create`` hard-fails (exit 1, no issue) if ANY ``--label``
        # does not already exist in the repo — one typo'd label from the
        # LLM would otherwise sink the whole file+claim. If the create
        # failed while we passed labels and the error is label-related,
        # retry ONCE with no labels: the issue (the point) still gets
        # filed, Talon stamps ``agent-claimed`` itself, and the dropped
        # labels are reported (never silently). The retry is still
        # ``gh issue create -R <repo> …`` so it stays inside the scoped
        # auto-approve seed — no new human gate.
        dropped_unknown_labels: List[str] = []
        label_retry = False
        if (not succeeded or m is None) and applied_labels:
            blob = (
                f"{stderr}\n{stdout}\n{shell_res.error or ''}"
            ).lower()
            if "label" in blob and (
                "not found" in blob
                or "could not add" in blob
                or "not a valid" in blob
                or "no label" in blob
            ):
                label_retry = True
                dropped_unknown_labels = list(applied_labels)
                applied_labels = []
                shell_res, succeeded, stdout, stderr, m = (
                    await _run_create([])
                )

        if not succeeded or m is None:
            reason_code, hint = _gh_failure_reason(
                shell_res, stdout, stderr,
                succeeded=succeeded, parsed_url=m is not None,
            )
            shell_data = shell_res.data or {}
            shell_returncode = shell_data.get("returncode")
            shell_timed_out = bool(shell_data.get("timed_out"))
            await self._log_fac_outcome(
                decision="filing_failed",
                reason_code=reason_code,
                repo=repo_resolved,
                filed=False,
                dispatched=False,
                extra={
                    "shell_status": str(shell_res.status),
                    "shell_returncode": shell_returncode,
                    "shell_timed_out": shell_timed_out,
                    "label_retry": label_retry,
                    "dropped_unknown_labels": dropped_unknown_labels,
                },
            )
            return ToolResult.failed(
                f"talon_file_and_claim: {reason_code} — {hint}",
                data={
                    "filed": False,
                    "dispatched": False,
                    "reason_code": reason_code,
                    "remediation": hint,
                    "repo": repo_resolved,
                    "shell_status": str(shell_res.status),
                    "shell_error": shell_res.error,
                    "shell_returncode": shell_returncode,
                    "shell_timed_out": shell_timed_out,
                    "stderr_tail": stderr[-300:],
                    "stdout_tail": stdout[-300:],
                    "label_retry": label_retry,
                    "dropped_unknown_labels": dropped_unknown_labels,
                },
            )

        issue_url = m.group(0)
        issue_number = int(m.group(1))

        claim = await self.talon_claim(repo=repo_resolved, issue=issue_number)
        claim_data = claim.data or {}
        dispatched = bool(claim_data.get("dispatched"))
        # Accept all three identifier shapes the dispatch paths produce:
        # CLI-background → ``job_id``; A2A (post-#1368) → ``task_id``;
        # historical mesh path (deleted) used ``message_id`` — left in
        # the lookup chain only for any stale serialized state that
        # could still surface during the rollout window.
        job_id = (
            claim_data.get("job_id")
            or claim_data.get("task_id")
            or claim_data.get("message_id")
            or None
        )
        result_data = {
            "filed": True,
            "issue_url": issue_url,
            "issue_number": issue_number,
            "repo": repo_resolved,
            "dispatched": dispatched,
            "job_id": job_id,
            "applied_labels": applied_labels,
            "stripped_labels": stripped_labels,
            "dropped_unknown_labels": dropped_unknown_labels,
            "label_retry": label_retry,
            "claim": claim_data,
        }
        notes = []
        if stripped_labels:
            notes.append(
                f"dropped Talon-reserved label(s) {stripped_labels} so the "
                f"claim wasn't pre-empted"
            )
        if dropped_unknown_labels:
            notes.append(
                f"refiled without unknown/invalid label(s) "
                f"{dropped_unknown_labels} (not present in the repo) so the "
                f"loop still closed"
            )
        stripped_note = f" ({'; '.join(notes)})" if notes else ""
        if dispatched:
            await self._log_fac_outcome(
                decision="filed_and_dispatched",
                reason_code="OK",
                repo=repo_resolved,
                filed=True,
                dispatched=True,
                issue_number=issue_number,
                job_id=job_id,
            )
            return ToolResult.ok(
                confirmation=(
                    f"Filed {repo_resolved}#{issue_number} ({issue_url}) "
                    f"and dispatched it to Talon (job_id={job_id})"
                    f"{stripped_note}."
                ),
                data=result_data,
            )
        # Issue filed but dispatch didn't take — surface as PARTIAL with
        # a discriminated reason code so the operator can tell "issue is
        # live, please claim by hand" from "everything broke." This is
        # the asymmetric-outcome case the partial-only-when-both-layers-
        # attempted rule was written for.
        result_data["reason_code"] = "DISPATCH_FAILED"
        result_data["remediation"] = (
            "the GitHub issue exists; call talon_claim(repo, issue) "
            "manually or investigate why the dispatch path returned "
            "dispatched=False."
        )
        await self._log_fac_outcome(
            decision="filed_dispatch_failed",
            reason_code="DISPATCH_FAILED",
            repo=repo_resolved,
            filed=True,
            dispatched=False,
            issue_number=issue_number,
        )
        return ToolResult.partial(
            confirmation=(
                f"Filed {repo_resolved}#{issue_number} ({issue_url}) but "
                f"the Talon dispatch did not take."
            ),
            error=(
                "talon_file_and_claim: DISPATCH_FAILED — "
                + (claim.error or "talon_claim did not report dispatched")
            ),
            data=result_data,
        )

    # ------------------------------------------------------------------
    # Internal: outcome audit row
    # ------------------------------------------------------------------

    async def _log_fac_outcome(
        self,
        *,
        decision: str,
        reason_code: str,
        repo: str,
        filed: bool,
        dispatched: bool,
        issue_number: Optional[int] = None,
        job_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an after-execution outcome row to ``security_audit_log``.

        The gate's pre-execution row records only the DECISION
        (auto_mode_allowed). #1383 demands that the OUTCOME of the tool
        run be recorded too — without that, an operator scanning the
        audit log can see "allowed" but not "what actually happened."

        Writes are best-effort: a missing SecurityFeature must not
        crash the loop-closing primitive. We log warnings instead.
        """
        security = self._get_security_feature()
        store = getattr(security, "permission_store", None) if security else None
        if store is None or not hasattr(store, "log_decision"):
            logger.debug(
                "talon_file_and_claim outcome not audited "
                f"(SecurityFeature/permission_store unavailable): "
                f"decision={decision} reason={reason_code} repo={repo}"
            )
            return

        summary: Dict[str, Any] = {
            "reason_code": reason_code,
            "repo": repo,
            "filed": filed,
            "dispatched": dispatched,
        }
        if issue_number is not None:
            summary["issue_number"] = issue_number
        if job_id is not None:
            summary["job_id"] = job_id
        if extra:
            summary.update(extra)
        try:
            await store.log_decision(
                feature_name="talon_feature",
                tool_name="talon_file_and_claim.outcome",
                action="tool_outcome",
                decision=decision,
                user_choice=None,
                args_summary=json.dumps(summary, default=str)[:1000],
            )
        except Exception as e:  # noqa: BLE001 — never break the caller
            logger.warning(
                f"Failed to write talon_file_and_claim outcome audit "
                f"row (reason_code={reason_code}): {e}"
            )

    @tool(
        name="talon_get_config",
        description=(
            "Read Talon runtime policy and mutable preference. This is the "
            "agent's control surface for its coding-agent backend/model, "
            "separate from normal chat LLM routing."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!talon config",
    )
    async def talon_get_config(self) -> ToolResult:
        """Return effective Talon policy and preference."""
        try:
            policy, preference = load_talon_policy_preference()
        except TalonRuntimeError as e:
            return ToolResult.failed(
                str(e),
                data={"success": False, "error": str(e)},
            )
        return ToolResult.ok(
            confirmation=(
                "Talon config loaded "
                f"(default={preference.default_backend}/{preference.default_model})"
            ),
            data={
                "success": True,
                "policy": asdict(policy),
                "preference": asdict(preference),
            },
        )

    @tool(
        name="talon_set_config",
        description=(
            "Update mutable Talon preferences only: default backend/model, "
            "auth lane, iterations, turns, clarification, and self-review. "
            "Operator policy is not changed by this tool."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!talon set-config",
    )
    async def talon_set_config(
        self,
        default_backend: Optional[str] = None,
        default_model: Optional[str] = None,
        default_auth_lane: Optional[str] = None,
        max_iterations: Optional[int] = None,
        max_turns: Optional[int] = None,
        skip_clarification: Optional[bool] = None,
        self_review: Optional[bool] = None,
    ) -> ToolResult:
        """Persist Talon preference updates under ``[talon.preference]``."""
        updates = {
            "default_backend": default_backend,
            "default_model": default_model,
            "default_auth_lane": default_auth_lane,
            "max_iterations": max_iterations,
            "max_turns": max_turns,
            "skip_clarification": skip_clarification,
            "self_review": self_review,
        }
        try:
            result = write_talon_preference(updates)
        except TalonRuntimeError as e:
            return ToolResult.failed(
                str(e),
                data={"success": False, "error": str(e)},
            )
        return ToolResult.ok(
            confirmation="Talon preference updated",
            data={"success": True, **result},
        )

    @tool(
        name="talon_verify",
        description=(
            "Reviewer-side audited test verification. Run one or more "
            "test commands and report a precise result state per command "
            "(passed / failed / blocked_by_policy / blocked_by_user / "
            "blocked_by_sandbox / tooling_error). Allowlisted project test "
            "commands (e.g. `uv run pytest ...`) run without prompting; "
            "anything else is approval-gated. Use this instead of ad-hoc "
            "shell so test evidence is structured and audited, and so a "
            "sandbox/policy block is never mislabeled as a user denial."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!talon verify",
    )
    async def talon_verify(
        self,
        commands: str,
        repo: str = "self",
        cwd: Optional[str] = None,
        ref: str = "",
        timeout: int = 600,
        note: str = "",
    ) -> ToolResult:
        """Run targeted verification commands and report structured evidence.

        Args:
            commands: One command per line (blank lines ignored).
                Allowlisted test runners run directly; non-allowlisted
                commands require approval and the block reason is recorded
                precisely.
            repo: ``owner/name`` (or ``self``) — used to locate the
                workspace clone to run in when ``cwd`` is not given.
            cwd: Working directory override. Defaults to the repo's
                workspace clone if present, else the project directory.
            ref: PR number / branch name / commit SHA to verify. When
                given, the workspace clone is fetched and checked out to
                that ref BEFORE any command runs, so a PR is verified
                against the PR's code and not whatever happened to be
                checked out (e.g. ``main``). A PR number may be written
                ``1630``, ``#1630``, or ``pr/1630``. If the fetch/checkout
                fails, no command runs and the result state is
                ``tooling_error`` — never a misleading test failure
                against the un-switched tree (issue #1631).
            timeout: Per-command wall-clock cap in seconds.
            note: Optional reviewer note included in the evidence (e.g.
                "local tests could not run; CI is the remaining hard gate").

        Returns:
            ``ToolResult.ok`` when every command passed; ``partial`` when
            any command failed, was blocked, or could not run (the tool
            itself ran fine — the LLM must not claim a clean pass).
            ``data`` carries the full ``VerificationEvidence`` dict plus
            ``repo``, ``cwd``, ``requested_ref``, ``checked_out_ref`` and
            ``head_sha`` so a reviewer can tell exactly which ref was
            verified, and ``confirmation`` carries the markdown evidence
            block for review/merge notes.
        """
        command_list = [c.strip() for c in str(commands).splitlines() if c.strip()]
        if not command_list:
            return ToolResult.failed(
                "talon_verify: no commands provided (one per line).",
                data={"success": False, "overall_state": "not_run"},
            )

        try:
            run_cwd = self._resolve_verify_cwd(repo, cwd)
        except ValueError as e:
            return ToolResult.failed(
                str(e), data={"success": False, "overall_state": "not_run"}
            )

        requested_ref = (ref or "").strip()
        if requested_ref:
            checkout = await self._git_checkout_ref(run_cwd, requested_ref)
            if not checkout.get("ok"):
                # Fetch/checkout failed: refuse to run the requested
                # commands against the un-switched tree. Reporting those
                # commands as ``tooling_error`` (not ``failed``) is the
                # whole point of #1631 — a reviewer must be able to tell a
                # ref-selection failure from a real code failure.
                reason = checkout.get("error") or (
                    f"could not check out ref {requested_ref!r}"
                )
                summary = (
                    f"requested ref {requested_ref!r} could not be checked out "
                    f"in {run_cwd}: {reason}. Commands were NOT run; this is "
                    "a tooling/ref-selection failure, not a code failure."
                )
                evidence = VerificationEvidence(
                    results=[
                        TestCommandResult(
                            command=cmd,
                            state=VerificationState.TOOLING_ERROR,
                            allowlisted=False,
                            summary=summary,
                        )
                        for cmd in command_list
                    ],
                    note=note,
                )
                data = {
                    "success": True,
                    "repo": self._resolve_repo(repo),
                    "cwd": str(run_cwd),
                    "requested_ref": requested_ref,
                    "checked_out_ref": None,
                    "head_sha": None,
                    **evidence.to_dict(),
                }
                return ToolResult.partial(
                    evidence.to_markdown(),
                    f"verification overall state: {evidence.overall_state.value}",
                    data=data,
                )

        verifier = TalonVerifier(
            execute=self._make_verify_executor(run_cwd),
            approve=self._make_verify_approver(repo, run_cwd),
        )
        try:
            timeout_int = max(1, int(timeout))
        except (TypeError, ValueError):
            timeout_int = 600

        evidence = await verifier.verify_commands(
            command_list, timeout=timeout_int, note=note
        )
        head = self._git_describe_head(run_cwd)
        data = {
            "success": True,
            "repo": self._resolve_repo(repo),
            "cwd": str(run_cwd),
            "requested_ref": requested_ref or None,
            "checked_out_ref": head.get("ref"),
            "head_sha": head.get("head_sha"),
            **evidence.to_dict(),
        }
        confirmation = evidence.to_markdown()
        if evidence.all_passed:
            return ToolResult.ok(confirmation=confirmation, data=data)
        return ToolResult.partial(
            confirmation,
            f"verification overall state: {evidence.overall_state.value}",
            data=data,
        )

    def _resolve_verify_cwd(self, repo: str, cwd: Optional[str]) -> Path:
        """Pick the directory to run verification commands in.

        ``cwd`` wins when given. Otherwise prefer the repo's workspace
        clone (the sandboxed checkout Talon worked in); fall back to the
        project directory. Raises ``ValueError`` if the resolved path is
        not a directory.
        """
        if cwd:
            path = Path(cwd).expanduser().resolve()
            if not path.is_dir():
                raise ValueError(f"talon_verify: cwd is not a directory: {path}")
            return path
        repo_resolved = self._resolve_repo(repo)
        workspace = self._workspace_path_for(repo_resolved)
        if (workspace / ".git").exists():
            return workspace
        from kestrel_sovereign.paths import project_dir

        return project_dir()

    def _make_verify_executor(self, run_cwd: Path):
        """Build an async executor that runs a command in ``run_cwd``.

        Returns a :class:`CommandExecution` whose ``ran`` flag records
        whether the process actually executed, so the verifier can tell
        a real exit code from a tooling/sandbox failure.
        """

        async def _execute(command: str, *, timeout: int = 600) -> CommandExecution:
            try:
                argv = shlex.split(command)
            except ValueError as e:
                return CommandExecution(ran=False, error=f"unparseable command: {e}")
            if not argv:
                return CommandExecution(ran=False, error="empty command")

            env = dict(os.environ)
            started = time.monotonic()
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=str(run_cwd),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
            except FileNotFoundError as e:
                return CommandExecution(
                    ran=False, error=f"command not found: {argv[0]} ({e})"
                )
            except PermissionError as e:
                return CommandExecution(
                    ran=False,
                    sandbox_denied=True,
                    error=f"sandbox refused to execute {argv[0]}: {e}",
                )
            except Exception as e:  # noqa: BLE001
                return CommandExecution(ran=False, error=f"failed to launch: {e}")

            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                duration_ms = int((time.monotonic() - started) * 1000)
                return CommandExecution(
                    ran=False,
                    duration_ms=duration_ms,
                    error=f"command timed out after {timeout}s",
                )
            duration_ms = int((time.monotonic() - started) * 1000)
            return CommandExecution(
                ran=True,
                returncode=proc.returncode,
                stdout=stdout_b.decode(errors="replace"),
                stderr=stderr_b.decode(errors="replace"),
                duration_ms=duration_ms,
            )

        return _execute

    def _make_verify_approver(self, repo: str, run_cwd: Path):
        """Build an async approver for non-allowlisted verify commands.

        Returns ``None`` (fail-closed, reported as ``blocked_by_policy``)
        when no SecurityFeature/approval queue is available. An explicit
        operator DENY is reported by the approval queue as a non-user
        scope, so :func:`verification.classify_denial` never mislabels it
        as a user denial.
        """

        async def _approve(command: str) -> Optional[tuple[bool, str]]:
            security = self._get_security_feature()
            queue = getattr(security, "approval_queue", None) if security else None
            if queue is None:
                return None
            try:
                approved, scope = await queue.request_approval(
                    feature_name="talon",
                    tool_name="verify_command",
                    tool_args={
                        "command": command,
                        "repo": self._resolve_repo(repo),
                        "cwd": str(run_cwd),
                    },
                )
                return bool(approved), str(scope)
            except (TimeoutError, asyncio.TimeoutError):
                return False, "timeout"
            except Exception as e:  # noqa: BLE001
                logger.error(f"talon_verify approval failed: {e}", exc_info=True)
                return None

        return _approve

    @staticmethod
    def _parse_verify_ref(ref: str) -> tuple[str, str]:
        """Classify a verify ref into ``("pr", number)`` or ``("ref", value)``.

        A PR number may be written ``1630``, ``#1630``, or ``pr/1630``
        (case-insensitive, with ``/``, ``#`` or ``-`` as the separator).
        Everything else — a branch name or a commit SHA — is returned as a
        plain ``ref`` and resolved by ``git checkout`` directly.
        """
        r = (ref or "").strip()
        m = re.match(r"^(?:#|pr[/#-]?)(\d+)$", r, re.IGNORECASE)
        if m:
            return "pr", m.group(1)
        if r.isdigit():
            return "pr", r
        return "ref", r

    async def _git_checkout_ref(self, workspace: Path, ref: str) -> Dict[str, Any]:
        """Fetch and check out ``ref`` in the workspace clone.

        ``ref`` forms (see :meth:`_parse_verify_ref`):

          * a PR number — fetched from ``refs/pull/<n>/head`` and checked
            out detached, so a PR is verified against its own head commit.
          * a branch name — the remote is fetched, then ``git checkout``
            switches to (and, if needed, creates a local tracking branch
            for) it.
          * a commit SHA — fetched best-effort, then checked out detached.

        Returns ``{"ok": bool, "error": str, "checked_out_ref": str,
        "head_sha": str}``. A non-``ok`` result means the requested ref
        could not be materialised; the caller must NOT run verification
        against the un-switched tree (issue #1631).
        """
        if not (workspace / ".git").exists():
            return {
                "ok": False,
                "error": (
                    f"{workspace} is not a git checkout; cannot select a "
                    "ref to verify"
                ),
            }
        kind, value = self._parse_verify_ref(ref)
        if not value:
            return {"ok": False, "error": "empty ref"}

        if kind == "pr":
            fetch = await self._git_run(
                ["fetch", "origin", f"refs/pull/{value}/head"],
                cwd=workspace,
                timeout=120,
            )
            if not fetch.get("ok"):
                return {
                    "ok": False,
                    "error": (
                        f"git fetch of PR #{value} failed: "
                        f"{fetch.get('error') or 'unknown error'}"
                    ),
                }
            checkout = await self._git_run(
                ["checkout", "--force", "--detach", "FETCH_HEAD"],
                cwd=workspace,
                timeout=60,
            )
            if not checkout.get("ok"):
                return {
                    "ok": False,
                    "error": (
                        f"git checkout of PR #{value} failed: "
                        f"{checkout.get('error') or 'unknown error'}"
                    ),
                }
        else:
            # Branch or SHA. Fetch the remote so newly-pushed branches
            # and commits are available locally. If a remote branch exists,
            # make origin/<branch> the source of truth so a stale local
            # branch cannot be verified accidentally.
            fetch = await self._git_run(
                ["fetch", "--all", "--prune"], cwd=workspace, timeout=120
            )
            if not fetch.get("ok"):
                return {
                    "ok": False,
                    "error": (
                        f"git fetch before checking out ref {value!r} failed: "
                        f"{fetch.get('error') or 'unknown error'}"
                    ),
                }
            remote_branch = await self._git_run(
                ["rev-parse", "--verify", f"refs/remotes/origin/{value}"],
                cwd=workspace,
                timeout=30,
            )
            if remote_branch.get("ok"):
                checkout = await self._git_run(
                    ["checkout", "--force", "-B", value, f"origin/{value}"],
                    cwd=workspace,
                    timeout=60,
                )
                if not checkout.get("ok"):
                    return {
                        "ok": False,
                        "error": (
                            f"git checkout of remote branch {value!r} failed: "
                            f"{checkout.get('error') or 'unknown error'}"
                        ),
                    }
            else:
                checkout = await self._git_run(
                    ["checkout", "--force", value], cwd=workspace, timeout=60
                )
                if not checkout.get("ok"):
                    return {
                        "ok": False,
                        "error": (
                            f"git checkout of ref {value!r} failed: "
                            f"{checkout.get('error') or 'unknown error'}"
                        ),
                    }

        head = self._git_describe_head(workspace)
        return {
            "ok": True,
            "checked_out_ref": head.get("ref"),
            "head_sha": head.get("head_sha"),
        }

    async def _git_run(
        self, args: List[str], *, cwd: Path, timeout: int = 120
    ) -> Dict[str, Any]:
        """Run a git subcommand in ``cwd`` and capture its outcome.

        Prefers the sanitized, token-bearing subprocess env so fetches
        from private remotes authenticate; falls back to the plain
        environment when no GitHub token is configured, so verifying a
        local branch never requires a token. Returns
        ``{"ok", "error", "stdout"}``.
        """
        try:
            env = self._build_subprocess_env()
        except RuntimeError:
            env = dict(os.environ)
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", *args,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as e:
            return {"ok": False, "error": f"git not found: {e}"}
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "ok": False,
                "error": f"git {args[0]} timed out after {timeout}s",
            }
        if proc.returncode != 0:
            return {
                "ok": False,
                "error": stderr_b.decode(errors="replace")[-500:]
                or f"git {args[0]} failed (exit {proc.returncode})",
            }
        return {"ok": True, "stdout": stdout_b.decode(errors="replace")}

    @staticmethod
    def _git_describe_head(workspace: Path) -> Dict[str, Optional[str]]:
        """Best-effort current ref + full HEAD SHA of a checkout.

        ``ref`` is the symbolic branch name when on a branch, else the
        short SHA (detached HEAD). ``head_sha`` is the full HEAD commit
        SHA. Either may be ``None`` if git can't be queried.
        """

        def _run(args: List[str]) -> Optional[str]:
            try:
                proc = subprocess.run(
                    ["git", *args],
                    cwd=str(workspace),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                return None
            if proc.returncode != 0:
                return None
            out = proc.stdout.strip()
            return out or None

        head_sha = _run(["rev-parse", "HEAD"])
        ref = _run(["rev-parse", "--abbrev-ref", "HEAD"])
        if ref == "HEAD":  # detached
            ref = _run(["rev-parse", "--short", "HEAD"])
        return {"ref": ref, "head_sha": head_sha}

    @tool(
        name="talon_workspace_status",
        description=(
            "Read-only: report on the talon workspace clone for a "
            "repo (path, exists, git HEAD, clean state, last fetch). "
            "No side effects."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!talon workspace-status",
    )
    async def talon_workspace_status(self, repo: str) -> ToolResult:
        """Inspect the talon workspace for ``repo``.

        Use this BEFORE ``talon_claim`` to verify the sandbox is
        ready: workspace exists, has a ``.git``, working tree is
        clean, last fetch was recent. If ``exists`` is False, call
        ``talon_setup_workspace`` first.
        """
        repo_resolved = self._resolve_repo(repo)
        state = self._workspace_state(repo_resolved)
        data = {"success": True, "repo": repo_resolved, **state}

        # Honesty: this is a read-only inspect, but reporting OK on
        # an unprovisioned/unsafe workspace would let the agent
        # narrate "workspace ready" off a state that talon_claim will
        # immediately reject. Surface as PARTIAL with the failing
        # condition so the LLM has to speak it.
        if state.get("safe") is False:
            return ToolResult.partial(
                confirmation=f"Read workspace state for {repo_resolved}",
                error=state.get("unsafe_reason") or "workspace path is unsafe",
                data=data,
            )
        if not state.get("exists") or not state.get("is_git"):
            return ToolResult.partial(
                confirmation=f"Read workspace state for {repo_resolved}",
                error=(
                    f"workspace at {state.get('path')} is not provisioned "
                    "(exists=False or no .git); call talon_setup_workspace "
                    "before talon_claim"
                ),
                data=data,
            )
        return ToolResult.ok(
            confirmation=(
                f"Workspace ready: {repo_resolved} at {state.get('path')} "
                f"(head={state.get('head')}, clean={state.get('clean')})"
            ),
            data=data,
        )

    @tool(
        name="talon_setup_workspace",
        description=(
            "Provision (or refresh) a sandboxed talon workspace clone "
            "for a repo. The clone lives outside the running agent's "
            "source tree, so talon_claim can operate without ever "
            "touching the agent's own checkout. Approval-gated."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!talon setup-workspace",
    )
    async def talon_setup_workspace(
        self,
        repo: str,
        fetch: bool = True,
    ) -> ToolResult:
        """Clone ``repo`` into the canonical talon workspace path.

        If the workspace already exists as a git checkout and
        ``fetch`` is True, runs ``git fetch --all --prune`` to
        refresh remote refs. Approval-gated: this is a network +
        disk operation that creates persistent state under the
        agent's home directory.

        Args:
            repo: GitHub repo in ``owner/name`` format, or ``"self"``.
            fetch: If True (default), fetch remote refs after clone
                or against an existing clone.

        Returns:
            ``{success, state: "created" | "refreshed" | "exists",
              workspace: {...}}`` on success. Failure returns
            ``{success: False, error, ...}``.
        """
        repo_resolved = self._resolve_repo(repo)
        workspace = self._workspace_path_for(repo_resolved)

        unsafe_reason = self._assert_workspace_safe(workspace)
        if unsafe_reason:
            return ToolResult.failed(
                unsafe_reason,
                data={"success": False, "state": "unsafe_workspace", "error": unsafe_reason},
            )

        approved = await self._request_workspace_approval(
            repo=repo_resolved, workspace=workspace, fetch=fetch,
        )
        if not approved:
            return ToolResult.failed(
                "Workspace setup not approved",
                data={
                    "success": False,
                    "state": "approval_denied",
                    "error": "Workspace setup not approved",
                    "workspace": str(workspace),
                },
            )

        existing = self._workspace_state(repo_resolved)
        if existing["exists"] and existing["is_git"]:
            if fetch:
                fetch_result = await self._git_fetch(workspace)
                if not fetch_result["ok"]:
                    return ToolResult.failed(
                        fetch_result["error"],
                        data={
                            "success": False,
                            "state": "fetch_failed",
                            "error": fetch_result["error"],
                            "workspace": self._workspace_state(repo_resolved),
                        },
                    )
                return ToolResult.ok(
                    confirmation=f"Refreshed workspace for {repo_resolved} (git fetch)",
                    data={
                        "success": True,
                        "state": "refreshed",
                        "workspace": self._workspace_state(repo_resolved),
                    },
                )
            return ToolResult.ok(
                confirmation=f"Workspace already exists for {repo_resolved} (no fetch)",
                data={
                    "success": True,
                    "state": "exists",
                    "workspace": existing,
                },
            )

        # Need to clone. Make sure the parent dir exists.
        try:
            workspace.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return ToolResult.failed(
                str(e),
                data={"success": False, "state": "mkdir_failed", "error": str(e)},
            )

        clone_url = f"https://github.com/{repo_resolved}.git"
        clone_result = await self._git_clone(clone_url, workspace)
        if not clone_result["ok"]:
            return ToolResult.failed(
                clone_result["error"],
                data={
                    "success": False,
                    "state": "clone_failed",
                    "error": clone_result["error"],
                },
            )

        return ToolResult.ok(
            confirmation=f"Created workspace for {repo_resolved} at {workspace}",
            data={
                "success": True,
                "state": "created",
                "workspace": self._workspace_state(repo_resolved),
            },
        )

    def _get_security_feature(self):
        agent = getattr(self, "agent", None)
        if agent is None:
            return None
        if hasattr(agent, "get_feature"):
            feat = agent.get_feature("SecurityFeature")
            if feat is not None:
                return feat
        features = getattr(agent, "features", None)
        if isinstance(features, dict):
            return features.get("SecurityFeature") or features.get("security")
        return None

    async def _request_workspace_approval(
        self, repo: str, workspace: Path, fetch: bool,
    ) -> bool:
        """Gate workspace creation/refresh through SecurityFeature.

        Setting up a workspace creates persistent state and pulls
        from the network — both worth a user-visible approval gate.
        Without an approval queue, dispatch is denied (fail-closed).
        """
        security = self._get_security_feature()
        if security is None or not hasattr(security, "approval_queue"):
            logger.warning(
                "SecurityFeature unavailable; talon_setup_workspace denied"
            )
            return False
        try:
            approved, _scope = await security.approval_queue.request_approval(
                feature_name="talon",
                tool_name="setup_workspace",
                tool_args={
                    "repo": repo,
                    "workspace_path": str(workspace),
                    "fetch": fetch,
                    "operation": (
                        "refresh existing clone"
                        if workspace.exists()
                        else "clone fresh"
                    ),
                },
            )
            return bool(approved)
        except (TimeoutError, asyncio.TimeoutError):
            return False
        except Exception as e:
            logger.error(f"Workspace approval failed: {e}", exc_info=True)
            return False

    async def _git_clone(self, url: str, dest: Path) -> Dict[str, Any]:
        try:
            env = self._build_subprocess_env()
        except RuntimeError as e:
            return {"ok": False, "error": str(e)}
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", url, str(dest),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=300,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"ok": False, "error": "git clone timed out (300s)"}
        if proc.returncode != 0:
            return {
                "ok": False,
                "error": stderr.decode(errors="replace")[-500:] or "git clone failed",
            }
        return {"ok": True}

    async def _git_fetch(self, workspace: Path) -> Dict[str, Any]:
        try:
            env = self._build_subprocess_env()
        except RuntimeError as e:
            return {"ok": False, "error": str(e)}
        proc = await asyncio.create_subprocess_exec(
            "git", "fetch", "--all", "--prune",
            cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=120,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"ok": False, "error": "git fetch timed out (120s)"}
        if proc.returncode != 0:
            return {
                "ok": False,
                "error": stderr.decode(errors="replace")[-500:] or "git fetch failed",
            }
        return {"ok": True}

    @staticmethod
    def _resolve_repo(repo: str) -> str:
        if repo and repo.lower() == "self":
            return os.environ.get(
                "GITHUB_SELF_REPO", "KestrelSovereignAI/kestrel-sovereign"
            )
        return repo

    @staticmethod
    def _workspace_root() -> Path:
        """Where workspace clones live.

        Default: ``~/.kestrel/talon_workspaces/``. Override with
        ``KESTREL_TALON_WORKSPACE_ROOT``. Always OUTSIDE the running
        agent's source tree by construction.
        """
        override = os.environ.get("KESTREL_TALON_WORKSPACE_ROOT")
        if override:
            return Path(override).expanduser().resolve()
        return (Path.home() / ".kestrel" / "talon_workspaces").resolve()

    @classmethod
    def _workspace_path_for(cls, repo: str) -> Path:
        """Canonical workspace clone path for a given repo string.

        Always under ``_workspace_root()`` and never inside the
        running agent's source tree — so even if a user sets
        ``KESTREL_TALON_WORKSPACE_ROOT`` to a path that *would*
        overlap, ``_assert_workspace_safe`` rejects it before any
        dispatch.
        """
        owner_repo = repo.replace("/", "__")  # filesystem-safe
        return cls._workspace_root() / owner_repo

    @classmethod
    def _assert_workspace_safe(cls, workspace: Path) -> Optional[str]:
        """Return None if the workspace is safe to operate against, or
        an error string explaining why it isn't.

        "Safe" means the workspace path is NOT the running agent's
        source root and NOT a directory containing it. This is
        structural, not a flag the agent can bypass.
        """
        if _path_contains(workspace, _RUNNING_AGENT_SOURCE_ROOT):
            return (
                f"Workspace path {workspace} contains the running agent's "
                f"source tree {_RUNNING_AGENT_SOURCE_ROOT}. The dispatcher "
                "refuses to operate on the agent's own source — Talon "
                "would commit/branch/quality-check against the running "
                "process. Use code_edit / propose_improvement for "
                "constitutional self-modification, or set "
                "KESTREL_TALON_WORKSPACE_ROOT to a path outside the "
                "source tree."
            )
        if _path_contains(_RUNNING_AGENT_SOURCE_ROOT, workspace):
            return (
                f"Workspace path {workspace} is inside the running agent's "
                f"source tree {_RUNNING_AGENT_SOURCE_ROOT}. Move the "
                "workspace root outside the source tree (set "
                "KESTREL_TALON_WORKSPACE_ROOT)."
            )
        return None

    @classmethod
    def _workspace_state(cls, repo_resolved: str) -> Dict[str, Any]:
        """Read-only snapshot of a workspace clone's state.

        Returns a dict with: ``path``, ``exists``, ``is_git`` (has
        ``.git``), ``head`` (current ref or ``None``), ``clean``
        (no uncommitted changes; ``None`` when not a git checkout),
        ``last_fetch_at`` (mtime of ``.git/FETCH_HEAD`` or ``None``),
        ``safe`` (passes ``_assert_workspace_safe``).
        """
        path = cls._workspace_path_for(repo_resolved)
        unsafe_reason = cls._assert_workspace_safe(path)

        state: Dict[str, Any] = {
            "repo": repo_resolved,
            "path": str(path),
            "exists": path.is_dir(),
            "is_git": False,
            "head": None,
            "clean": None,
            "last_fetch_at": None,
            "safe": unsafe_reason is None,
        }
        if unsafe_reason:
            state["unsafe_reason"] = unsafe_reason
            return state

        if not path.is_dir():
            return state

        git_dir = path / ".git"
        state["is_git"] = git_dir.exists()
        if not state["is_git"]:
            return state

        # HEAD ref
        head_file = git_dir / "HEAD"
        if head_file.is_file():
            try:
                head_text = head_file.read_text(encoding="utf-8").strip()
                if head_text.startswith("ref: refs/heads/"):
                    state["head"] = head_text[len("ref: refs/heads/"):]
                else:
                    state["head"] = head_text[:40]
            except OSError:
                pass

        # Working-tree cleanliness — porcelain check via subprocess so
        # we don't reimplement git status. Tolerates missing git.
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            state["clean"] = (proc.returncode == 0 and not proc.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            state["clean"] = None

        fetch_head = git_dir / "FETCH_HEAD"
        if fetch_head.is_file():
            try:
                state["last_fetch_at"] = datetime.fromtimestamp(
                    fetch_head.stat().st_mtime, tz=timezone.utc,
                ).isoformat()
            except OSError:
                pass

        return state

    @tool(
        name="talon_batch",
        description="Dispatch a batch of issues to Talon (by label or PRD).",
        category=ToolCategory.UTILITY,
        command_prefix="!talon batch",
    )
    async def talon_batch(
        self,
        repo: str,
        label: str = "",
        prd: str = "",
    ) -> ToolResult:
        """Dispatch batch processing to Talon.

        Args:
            repo: GitHub repo in owner/name format.
            label: Filter issues by this label.
            prd: Path to a PRD JSON file for batch mode.
        """
        if prd:
            cli_result = await self._dispatch_via_cli_background(
                ["batch", "--prd", prd],
                label=f"batch:prd={prd}",
                extra_meta={"prd": prd},
            )
        elif label:
            repo_resolved = self._resolve_repo(repo)
            cli_result = await self._dispatch_via_cli_background(
                ["batch", "--repo", repo_resolved, "--label", label],
                label=f"batch:{repo_resolved}:label={label}",
                extra_meta={"repo": repo_resolved, "github_label": label},
            )
        else:
            return ToolResult.failed(
                "Provide either label or prd",
                data={"dispatched": False, "error": "Provide either label or prd"},
            )

        if cli_result.get("dispatched"):
            return ToolResult.ok(
                confirmation=f"Dispatched batch (job_id={cli_result.get('job_id', '?')})",
                data=cli_result,
            )
        return ToolResult.failed(
            cli_result.get("error") or "talon batch dispatch failed",
            data=cli_result,
        )

    @tool(
        name="talon_status",
        description="Check status of Talon jobs (running, completed, failed).",
        category=ToolCategory.UTILITY,
        command_prefix="!talon status",
    )
    async def talon_status(self) -> ToolResult:
        """Check status of dispatched Talon jobs.

        Reaps any background CLI subprocess that has finished since
        the last call (updates ``status`` to ``complete`` or
        ``failed`` based on returncode), then summarises.
        """
        # Pull in any CLI jobs persisted before a restart so they are
        # visible again even without an in-memory process handle.
        self._reload_persisted_jobs()

        # Reap finished background CLI jobs
        jobs_changed = False
        for jid, info in list(self._jobs.items()):
            if self._reap_cli_job(info):
                jobs_changed = True

        if jobs_changed:
            self._persist_jobs()

        # Reconcile A2A-dispatched jobs against Talon's task_store.
        # Mesh used to do this via inbox polling for complete/reject
        # messages; the A2A equivalent is querying the recipient's
        # task by id and mapping its TaskState back to coordinator
        # status. Without this, talon_status reports method=a2a jobs
        # as "dispatched" forever and downstream pollers
        # (FeatureFeature implementation gate, workflows runner) hang
        # waiting for a completion that never surfaces. Codex P2 on
        # PR #1368.
        host_url = self._discover_host_url()
        a2a_jobs_to_check = [
            (jid, info) for jid, info in self._jobs.items()
            if info.get("method") == "a2a"
            and info.get("status") in ("dispatched", "running")
        ]
        if host_url and a2a_jobs_to_check:
            a2a_changed = False
            for jid, info in a2a_jobs_to_check:
                if await self._reconcile_a2a_job(jid, info, host_url):
                    a2a_changed = True
            if a2a_changed:
                self._persist_jobs()

        def _public(info: Dict[str, Any]) -> Dict[str, Any]:
            # Strip non-serialisable fields (the asyncio Process handle).
            return {
                k: v for k, v in info.items()
                if k != "process"
            }

        running = [
            {**_public(info), "id": jid}
            for jid, info in self._jobs.items()
            if info.get("status") in ("dispatched", "running")
        ]
        done = [
            {**_public(info), "id": jid}
            for jid, info in self._jobs.items()
            if info.get("status") in (
                "complete", "failed", "reject", "finished_unknown",
            )
        ]

        data = {
            "running": len(running),
            "completed": len(done),
            "jobs": running + done,
        }
        return ToolResult.ok(
            confirmation=(
                f"Talon jobs: running={len(running)}, completed={len(done)}"
            ),
            data=data,
        )

    @tool(
        name="talon_job_log",
        description=(
            "Tail the log file of a dispatched Talon job. Use the "
            "job_id returned by talon_claim. Read-only."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!talon job-log",
    )
    async def talon_job_log(
        self, job_id: str, lines: int = 200,
    ) -> ToolResult:
        """Return the last ``lines`` lines of a job's combined log."""
        # Reload persisted CLI jobs so logs are tail-able after restart.
        self._reload_persisted_jobs()
        info = self._jobs.get(job_id)
        if not info:
            return ToolResult.failed(
                f"Unknown job_id: {job_id}",
                data={"job_id": job_id},
            )

        log_path = info.get("log_path")
        if not log_path or not os.path.isfile(log_path):
            return ToolResult.failed(
                f"Log file missing: {log_path}",
                data={"job_id": job_id, "log_path": log_path},
            )

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-max(1, int(lines)):]
        except OSError as e:
            return ToolResult.failed(str(e), data={"job_id": job_id})

        return ToolResult.ok(
            confirmation=(
                f"Tailed {len(tail)} line(s) from {log_path} "
                f"(status={info.get('status')}, rc={info.get('returncode')})"
            ),
            data={
                "success": True,
                "job_id": job_id,
                "log_path": log_path,
                "status": info.get("status"),
                "returncode": info.get("returncode"),
                "lines": len(tail),
                "content": "".join(tail),
            },
        )

    @tool(
        name="talon_wait",
        description=(
            "Block the current turn until a specific Talon job reaches a "
            "terminal state (complete/failed/finished_unknown) or the "
            "timeout expires, polling the durable job registry instead "
            "of shelling out to `sleep`. Returns the terminal status, "
            "the return code when known, a log tail, and timeout "
            "metadata if the job is still running. Use this when "
            "actively supervising a job this turn; the generic wait "
            "reconciler cron (signal path) handles unattended completions. "
            "Pass mode='signal' to register a watch and return immediately "
            "instead of holding the turn (talon is already auto-monitored, "
            "so this is mostly for symmetry with the generic wait tool)."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!talon wait",
    )
    async def talon_wait(
        self,
        job_id: str,
        timeout_seconds: int = 600,
        poll_interval_seconds: int = 10,
        mode: str = "block",
    ) -> ToolResult:
        """Wait for a dispatched Talon job to finish.

        Args:
            job_id: The job_id returned by talon_claim.
            timeout_seconds: Maximum seconds to wait before returning
                still-running (capped at the enforced maximum).
            poll_interval_seconds: Seconds between registry polls.
            mode: ``"block"`` (default) holds the turn polling until the job
                is terminal; ``"signal"`` registers a watch and returns
                immediately, waking the agent via a signal on completion.
        """
        mode = str(mode).strip().lower() if mode else "block"
        if mode not in ("block", "signal"):
            return ToolResult.failed(
                f"mode must be 'block' or 'signal', got {mode!r}"
            )

        if mode == "signal":
            # Register a watch and return immediately. talon is already
            # auto-monitored via active_handles, so this is redundant for
            # talon specifically, but it keeps the interface symmetric and
            # still wakes the agent on completion.
            try:
                await register_wait_watch(self.agent, f"talon:{job_id}")
            except ValueError as exc:
                return ToolResult.failed(str(exc))
            return ToolResult.ok(
                confirmation=(
                    f"Watching talon:{job_id}; will wake on completion via "
                    f"the wait reconciler"
                ),
                data={"ref": f"talon:{job_id}", "mode": "signal", "watching": True},
            )

        # Thin wrapper over the generic wait engine: the TalonWaitable
        # provider runs the same reap/reconcile single-step the legacy
        # loop ran per iteration; the engine owns the loop, the cap, and
        # the ToolResult mapping. Name kept so existing callers work.
        return await run_wait_loop(
            TalonWaitable(self),
            job_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            max_seconds=self._TALON_WAIT_MAX_SECONDS,
        )

    @tool(
        name="talon_pause",
        description="Pause the autonomous Talon loop (kill switch).",
        category=ToolCategory.SYSTEM,
        command_prefix="!talon pause",
    )
    async def talon_pause(self) -> ToolResult:
        """Pause the Talon dispatch loop.

        Disables the signal_dispatch scheduler so no new work is
        dispatched until resumed with !talon resume.
        """
        scheduler = getattr(self.agent, '_scheduler', None)
        if scheduler and hasattr(scheduler, 'remove_schedule'):
            scheduler.remove_schedule("signal_dispatch")
            return ToolResult.ok(
                confirmation="Talon dispatch paused. Use !talon resume to restart.",
                data={"paused": True, "message": "Talon dispatch paused. Use !talon resume to restart."},
            )
        return ToolResult.failed(
            "Scheduler not available",
            data={"paused": False, "error": "Scheduler not available"},
        )

    @tool(
        name="talon_resume",
        description="Resume the autonomous Talon loop after pause.",
        category=ToolCategory.SYSTEM,
        command_prefix="!talon resume",
    )
    async def talon_resume(self) -> ToolResult:
        """Resume the Talon dispatch loop after a pause."""
        scheduler = getattr(self.agent, '_scheduler', None)
        if scheduler and hasattr(scheduler, 'add_schedule'):
            scheduler.add_schedule("signal_dispatch", "5 8 * * *", "signal_dispatch")
            return ToolResult.ok(
                confirmation="Talon dispatch resumed at 08:05 daily.",
                data={"resumed": True, "message": "Talon dispatch resumed at 08:05 daily."},
            )
        return ToolResult.failed(
            "Scheduler not available",
            data={"resumed": False, "error": "Scheduler not available"},
        )

    @tool(
        name="talon_health",
        description=(
            "Check whether kestrel-talon is reachable and runnable: "
            "binary discoverable, subprocess env clean, executes "
            "``--help`` successfully. Read-only, no dispatch."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!talon health",
    )
    async def talon_health(self) -> ToolResult:
        """Smoke-test the talon CLI path without dispatching real work.

        Returns a structured report covering:

        * ``binary``: where the executable lives (or why we couldn't
          find it).
        * ``env``: which Anthropic keys were stripped, and which
          GitHub token name was found. The actual token VALUE is
          never returned.
        * ``execute``: result of ``kestrel-talon --help``: returncode,
          first line of stdout, and stderr summary if it failed.

        Use this BEFORE ``talon_claim`` on a real issue — most past
        dispatch failures came from missing/wrong env, and this
        check catches them in seconds without burning a Claude Max
        session or touching GitHub.
        """
        report: Dict[str, Any] = {"healthy": False}

        def _wrap(r: Dict[str, Any]) -> ToolResult:
            """Final wrap: healthy → OK; any failure → ERROR with the
            most-specific reason in the error string so the LLM can't
            narrate "talon healthy" off a binary-not-found body."""
            if r.get("healthy"):
                return ToolResult.ok(
                    confirmation=(
                        f"kestrel-talon healthy "
                        f"(binary={r.get('binary', {}).get('path')})"
                    ),
                    data=r,
                )
            # Pull the most specific error available. Stage dicts may
            # carry ``error`` directly OR a non-zero ``returncode`` +
            # ``stderr_tail`` (the --help-fails-mid-startup path); both
            # need to surface, otherwise the agent loses the actual
            # cause that talon_health exists to diagnose.
            for stage in ("execute", "env", "binary"):
                stage_data = r.get(stage)
                if not isinstance(stage_data, dict):
                    continue
                if stage_data.get("error"):
                    return ToolResult.failed(
                        f"talon health {stage}: {stage_data['error']}",
                        data=r,
                    )
                if stage == "execute" and stage_data.get("ok") is False:
                    rc = stage_data.get("returncode")
                    stderr_tail = (stage_data.get("stderr_tail") or "").strip()
                    detail = stderr_tail[-300:] if stderr_tail else "no stderr"
                    return ToolResult.failed(
                        f"talon health execute: --help exited rc={rc}; "
                        f"stderr_tail={detail!r}",
                        data=r,
                    )
            return ToolResult.failed(
                "talon health check failed",
                data=r,
            )

        # 1. Binary discovery
        talon_bin = self._find_talon_bin()
        if not talon_bin:
            report["binary"] = {
                "found": False,
                "error": (
                    "kestrel-talon not found. Set KESTREL_TALON_BIN, "
                    "`uv sync` it into this venv, or place a sibling "
                    "checkout at ../kestrel-talon with its own .venv."
                ),
            }
            return _wrap(report)
        report["binary"] = {"found": True, "path": talon_bin}

        # 2. Env cleanliness
        stripped = [
            k for k in self._ANTHROPIC_KEYS_TO_STRIP if k in os.environ
        ]
        env_report: Dict[str, Any] = {
            "stripped_anthropic_keys": stripped,
        }
        try:
            built_env = self._build_subprocess_env()
        except RuntimeError as e:
            env_report["error"] = str(e)
            report["env"] = env_report
            return _wrap(report)

        token_source = next(
            (
                name
                for name in ("GH_TOKEN", "GITHUB_TOKEN", "GITHUB_PAT")
                if os.environ.get(name)
            ),
            None,
        )
        env_report["github_token_source"] = token_source
        env_report["env_var_count"] = len(built_env)
        report["env"] = env_report

        # 3. Execute --help to prove the binary is runnable AND that
        # talon's own startup (which imports Claude Agent SDK) doesn't
        # crash with our env. Bounded: --help returns in < 5s.
        cmd = [talon_bin, "--help"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=built_env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=15,
            )
        except asyncio.TimeoutError:
            report["execute"] = {
                "ok": False,
                "error": "kestrel-talon --help timed out after 15s",
            }
            return _wrap(report)
        except Exception as e:
            report["execute"] = {"ok": False, "error": str(e)}
            return _wrap(report)

        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")
        first_out_line = next((ln for ln in out.splitlines() if ln.strip()), "")
        report["execute"] = {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "first_line": first_out_line[:200],
            "stderr_tail": err[-400:] if err else "",
        }

        report["healthy"] = bool(report["execute"]["ok"])
        return _wrap(report)

    # ------------------------------------------------------------------
    # Internal: Mesh dispatch (preferred)
    # ------------------------------------------------------------------

    async def _dispatch_via_a2a(
        self, repo: str, issue_number: int, title: str = ""
    ) -> Dict[str, Any]:
        """Submit an A2A task to Talon — the replacement for mesh dispatch.

        Builds a ``TaskSendParams``-shaped payload and POSTs it to
        Talon's ``/api/agent/tasks/send`` endpoint. The receiving side
        creates the task, fires ``a2a.task_submitted`` so Talon's
        cognition loop wakes up, and acts on the assignment. Same
        functional contract as the prior mesh dispatch but with
        durable persistence (TaskStore vs in-memory inbox), lifecycle
        states (SUBMITTED → WORKING → COMPLETED), and signal-driven
        wakeup (#1366 / #1367)."""
        from uuid import uuid4

        host_url = self._discover_host_url()
        if not host_url:
            return {"dispatched": False, "reason": "no_a2a_host"}

        sender = getattr(self.agent, "agent_name", None) or getattr(
            self.agent, "did", "kestrel",
        )
        task_id = uuid4().hex
        session_id = uuid4().hex
        body = (
            f"Assignment: implement issue {repo}#{issue_number} "
            f"({title or 'no title'}). Take the work and report back "
            f"via task completion."
        )
        payload = json.dumps({
            "id": task_id,
            "sessionId": session_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": body}],
            },
            "metadata": {
                "sender": sender,
                "skill": "workflow.assign",
                "repo": repo,
                "issue_number": issue_number,
                "issue_title": title or f"#{issue_number}",
            },
        }).encode("utf-8")

        url = f"{host_url}/api/agents/talon/api/agent/tasks/send"
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            await asyncio.to_thread(
                lambda: urllib.request.urlopen(req, timeout=10).read()
            )
            self._jobs[task_id] = {
                "repo": repo, "issue": issue_number,
                "status": "dispatched", "method": "a2a",
            }
            return {
                "dispatched": True, "method": "a2a",
                "task_id": task_id, "repo": repo, "issue": issue_number,
            }
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            logger.debug(f"A2A dispatch failed: {e}")
            return {"dispatched": False, "reason": "a2a_unavailable", "error": str(e)}

    # ------------------------------------------------------------------
    # Internal: CLI fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _find_talon_bin() -> Optional[str]:
        """Locate the kestrel-talon executable.

        Search order:
        1. ``KESTREL_TALON_BIN`` env var (explicit override).
        2. ``shutil.which`` against the running process's PATH (works
           when ``uv sync`` installed kestrel-talon into this venv).
        3. Sibling-checkout convention used in dev: a ``kestrel-talon``
           directory next to ``kestrel-sovereign`` with its own
           ``.venv/bin/kestrel-talon``. Required because Jason's
           workflow keeps talon as a separate ``uv run`` source tree
           rather than installed into the kestrel venv.
        """
        override = os.environ.get("KESTREL_TALON_BIN")
        if override and os.path.isfile(override) and os.access(override, os.X_OK):
            return override

        on_path = shutil.which("kestrel-talon")
        if on_path:
            return on_path

        # Sibling layout: parents = talon, features, kestrel_sovereign,
        # kestrel-sovereign (project root), then projects/. The sibling
        # checkout lives next to the project root.
        sibling = (
            Path(__file__).resolve().parents[4]
            / "kestrel-talon"
            / ".venv"
            / "bin"
            / "kestrel-talon"
        )
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return str(sibling)

        return None

    # Anthropic credentials kestrel-talon must NOT see — its Claude
    # Agent SDK call chain auto-uses Claude Max OAuth from ``~/.claude``,
    # but only if no API-key env var is set. If kestrel-sovereign was
    # launched with ``ANTHROPIC_API_KEY`` (which it usually is — the
    # main agent uses it for its own LLM calls), passing that env
    # straight through silently flips talon onto API-key billing
    # AND breaks any "I am running as user X" identity assertions.
    #
    # See ``feedback_kestrel_talon.md``: "API key is specifically
    # stripped." Order matters too — Claude Agent SDK merges parent
    # ``os.environ`` after we hand it our env dict, so the talon
    # binary itself further mutates ``os.environ`` at runtime; we
    # only need to make sure our subprocess starts clean.
    _ANTHROPIC_KEYS_TO_STRIP = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_API_KEY",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
    )

    @staticmethod
    def _build_subprocess_env(
        backend: str = "claude",
        auth_lane: str = "oauth",
    ) -> Dict[str, str]:
        """Construct the env dict for a kestrel-talon subprocess.

        Backend-specific sanitization keeps Talon from inheriting credentials
        unrelated to the selected runtime, and verifies a GitHub token is
        present. Raises ``RuntimeError`` with an actionable message if a
        required var is missing; callers convert that into a structured
        ``dispatched=False`` response.
        """
        try:
            env, _stripped = sanitize_env_for_backend(
                normalize_backend(backend) or "claude",
                normalize_auth_lane(auth_lane) or "oauth",
            )
            return env
        except TalonRuntimeError as e:
            raise RuntimeError(str(e)) from e

    def _job_log_dir(self) -> Path:
        """Where job log files live.

        Per-agent under the agent's data directory when available
        (``<storage_path>/talon_jobs/``) so logs survive process
        restarts and aren't shared across agents in the multi_agent.
        Falls back to ``/tmp`` when the agent has no storage_path
        (test stubs, ephemeral runs).
        """
        storage_path = getattr(self.agent, "storage_path", None) if self.agent else None
        if storage_path:
            base = Path(storage_path).parent / "talon_jobs"
        else:
            base = Path("/tmp/kestrel_talon_jobs")
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _jobs_registry_path(self) -> Path:
        """Durable registry of CLI-background job metadata.

        Lives next to the per-agent log files so it survives Kestrel
        restarts (``<storage_path>/talon_jobs/jobs.json``).
        """
        return self._job_log_dir() / "jobs.json"

    def _job_exit_path(self, job_id: str) -> Path:
        """Sidecar file the dispatch wrapper writes its exit code to.

        Used after a Kestrel restart to recover the true exit status
        of CLI background jobs that finished while no in-process handle
        was awaiting them.
        """
        return self._job_log_dir() / f"{job_id}.exit"

    @staticmethod
    def _tail_job_log(path: Any, lines: int = 20) -> str:
        """Best-effort tail of a job's combined log file.

        Used by :class:`TalonWaitable` to attach a short context snippet to
        the COGNITION signal the reconciler emits, so the agent does not have to
        call ``talon_job_log`` as a follow-up tool just to see what
        happened. Returns ``""`` on any read error — the signal is
        still useful without it.
        """
        if not path:
            return ""
        try:
            with open(str(path), "r", encoding="utf-8") as f:
                buf = f.readlines()
        except (OSError, UnicodeDecodeError):
            return ""
        if not buf:
            return ""
        return "".join(buf[-lines:])

    def _reap_cli_job(self, info: Dict[str, Any]) -> bool:
        """Refresh a single ``cli_background`` job's status in place.

        Returns ``True`` when the job's persisted state changed so the
        caller knows to re-persist the registry. No-op (returns
        ``False``) for jobs that are not ``cli_background`` or are no
        longer in ``dispatched``/``running``. Shared by ``talon_status``
        and ``talon_wait`` so the reaping logic lives in one place.
        """
        if info.get("method") != "cli_background":
            return False
        if info.get("status") not in ("dispatched", "running"):
            return False
        proc = info.get("process")
        if proc is None:
            # Reloaded after restart: no live handle. The sidecar exit
            # file is the authoritative source of truth — if it exists,
            # the wrapper recorded the exit code. If not, fall back to
            # pid liveness (best-effort; PID reuse can produce a false
            # 'running').
            rc = self._read_exit_sidecar(info.get("exit_path"))
            if rc is not None:
                info["status"] = "complete" if rc == 0 else "failed"
                info["returncode"] = rc
                info.setdefault(
                    "completed_at",
                    datetime.now(timezone.utc).isoformat(),
                )
                return True
            if self._pid_alive(info.get("pid")):
                info["status"] = "running"
                return False
            # Process gone, no sidecar: status genuinely unknown. Do NOT
            # claim 'complete' — that would lie about failures that
            # exited before the wrapper wrote.
            info["status"] = "finished_unknown"
            info.setdefault("returncode", None)
            info.setdefault(
                "completed_at",
                datetime.now(timezone.utc).isoformat(),
            )
            return True
        rc = proc.returncode
        if rc is None:
            info["status"] = "running"
            return False
        info["status"] = "complete" if rc == 0 else "failed"
        info["returncode"] = rc
        info["completed_at"] = datetime.now(timezone.utc).isoformat()
        return True

    async def _reconcile_a2a_job(
        self, jid: str, info: Dict[str, Any], host_url: str
    ) -> bool:
        """Refresh a single ``a2a`` job's status against Talon's task_store.

        Queries the recipient's A2A task by id and maps its TaskState
        back to coordinator status: SUBMITTED/WORKING→running,
        COMPLETED→complete, FAILED/CANCELED→failed. Returns ``True`` when
        the persisted state changed so the caller knows to re-persist the
        registry. No-op (returns ``False``) for jobs that are not ``a2a``,
        are no longer ``dispatched``/``running``, or when the network
        query fails. Shared by ``talon_status`` and ``talon_wait`` so the
        A2A reconciliation lives in one place — mirroring ``_reap_cli_job``
        for the CLI transport.
        """
        if info.get("method") != "a2a":
            return False
        if info.get("status") not in ("dispatched", "running"):
            return False
        if not host_url:
            return False
        url = f"{host_url}/api/agents/talon/api/agent/tasks/{jid}"
        req = urllib.request.Request(
            url,
            method="GET",
            headers={"Content-Type": "application/json"},
        )
        try:
            raw = await asyncio.to_thread(
                lambda: urllib.request.urlopen(req, timeout=5).read()
            )
            task_payload = json.loads(raw)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
            # Network blip / Talon offline / task not yet visible to
            # Talon's task_store. Leave the coordinator's row in its
            # current state; the next poll will retry.
            return False
        state = (
            task_payload.get("status", {}) or {}
        ).get("state") or task_payload.get("status")
        # Map A2A TaskState → coordinator status. SUBMITTED and WORKING
        # stay "running"; COMPLETED→complete; FAILED/CANCELED→failed.
        if state in ("submitted", "working"):
            if info.get("status") != "running":
                info["status"] = "running"
                return True
            return False
        if state == "completed":
            info["status"] = "complete"
            info["completed_at"] = datetime.now(timezone.utc).isoformat()
            return True
        if state in ("failed", "canceled"):
            info["status"] = "failed"
            info["completed_at"] = datetime.now(timezone.utc).isoformat()
            err_msg = (
                (task_payload.get("status", {}) or {})
                .get("message", {}) or {}
            )
            info["error"] = err_msg.get("text") or state
            return True
        return False

    @staticmethod
    def _read_exit_sidecar(path: Any) -> Optional[int]:
        """Read an exit-code sidecar; ``None`` if absent or malformed."""
        if not path:
            return None
        try:
            with open(str(path), "r", encoding="utf-8") as f:
                content = f.read().strip()
        except (OSError, UnicodeDecodeError):
            return None
        if not content:
            return None
        try:
            return int(content)
        except ValueError:
            return None

    @staticmethod
    def _pid_alive(pid: Any) -> bool:
        """Return True if ``pid`` names a live process (signal 0 probe)."""
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            return False
        if pid_int <= 0:
            return False
        try:
            os.kill(pid_int, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but is owned by another user — still alive.
            return True
        except OSError:
            return False
        return True

    def _persist_jobs(self) -> None:
        """Write CLI-background job metadata to the durable registry.

        Excludes non-serialisable fields (the asyncio ``process``
        handle) so ``talon_status`` and ``talon_job_log`` keep working
        after a feature/server restart.
        """
        registry: Dict[str, Any] = {}
        for jid, info in self._jobs.items():
            if info.get("method") != "cli_background":
                continue
            registry[jid] = {
                k: v for k, v in info.items() if k != "process"
            }
        path = self._jobs_registry_path()
        # Use a unique tmp file per writer so concurrent _persist_jobs
        # calls from sibling feature instances cannot clobber each
        # other's pre-replace temp content. os.replace() is atomic on
        # POSIX so the final rename remains race-free.
        tmp_path: Optional[str] = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
            )
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(registry, f)
            os.replace(tmp_path, path)
            tmp_path = None
        except OSError as e:
            logger.warning(f"Failed to persist talon job registry: {e}")
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _reload_persisted_jobs(self) -> None:
        """Merge durably-persisted CLI jobs into the in-memory map.

        In-process jobs (which still hold a live ``process`` handle)
        win; only job_ids absent from memory are reloaded so a
        restarted feature regains status/log visibility.
        """
        try:
            path = self._jobs_registry_path()
        except (TypeError, OSError) as e:
            # Test stubs / discovery harness can construct the
            # feature with a Mock agent whose ``storage_path`` is
            # not a valid path. Skip reload silently in that case;
            # real agents always have a real storage_path.
            logger.debug("Talon job registry path unavailable: %s", e)
            return
        if not path.is_file():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to read talon job registry: {e}")
            return
        if not isinstance(registry, dict):
            return
        for jid, info in registry.items():
            if jid in self._jobs or not isinstance(info, dict):
                continue
            # Reloaded jobs have no live process handle.
            info.pop("process", None)
            self._jobs[jid] = info

    async def _dispatch_via_cli_background(
        self,
        args: List[str],
        label: str,
        env: Optional[Dict[str, str]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Launch kestrel-talon as a background subprocess and return.

        The subprocess writes combined stdout+stderr to a log file
        the caller can tail later via ``talon_job_log(job_id)``.
        Job metadata is tracked in ``self._jobs`` so ``talon_status``
        can reap finished processes and report them.
        """
        talon_bin = self._find_talon_bin()
        if not talon_bin:
            return {
                "dispatched": False,
                "error": (
                    "kestrel-talon not found. Set KESTREL_TALON_BIN, install "
                    "kestrel-talon into the kestrel-sovereign venv "
                    "(`uv sync`), or place a sibling checkout at "
                    "../kestrel-talon with its own .venv."
                ),
            }

        if env is None:
            try:
                env = self._build_subprocess_env()
            except RuntimeError as e:
                return {"dispatched": False, "error": str(e)}

        job_id = uuid.uuid4().hex
        log_path = self._job_log_dir() / f"{job_id}.log"
        exit_path = self._job_exit_path(job_id)
        cmd = [talon_bin] + args

        try:
            log_file = open(log_path, "w", encoding="utf-8")
        except OSError as e:
            return {"dispatched": False, "error": f"Cannot open log file: {e}"}

        # Wrap with sh -c so the exit code is written to a sidecar file
        # atomically when the subprocess terminates. The sidecar is the
        # authoritative source of truth for status after a Kestrel
        # restart, when no live process handle exists.
        exit_quoted = shlex.quote(str(exit_path))
        wrapper_script = (
            f'"$@"; rc=$?; printf "%s" "$rc" > {exit_quoted}.tmp && '
            f'mv {exit_quoted}.tmp {exit_quoted}; exit "$rc"'
        )
        wrapped = ["sh", "-c", wrapper_script, "_talon_wrapper"] + cmd

        try:
            proc = await asyncio.create_subprocess_exec(
                *wrapped,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                cwd=os.environ.get("KESTREL_TALON_CWD") or str(_DEFAULT_PROJECT_PARENT),
            )
        except Exception as e:
            log_file.close()
            return {"dispatched": False, "error": str(e)}
        # Close our handle — the child has its own. Otherwise the file
        # stays open in our process for the lifetime of the subprocess
        # and no one else can rotate/inspect it cleanly.
        log_file.close()

        info: Dict[str, Any] = {
            "method": "cli_background",
            "label": label,
            "command": " ".join(cmd),
            "status": "dispatched",
            "pid": proc.pid,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "log_path": str(log_path),
            "exit_path": str(exit_path),
            "process": proc,
        }
        if extra_meta:
            info.update(extra_meta)
        self._jobs[job_id] = info
        self._persist_jobs()

        logger.info(
            f"Dispatched talon job {job_id} (pid={proc.pid}, "
            f"label={label}, log={log_path})"
        )

        return {
            "dispatched": True,
            "method": "cli_background",
            "job_id": job_id,
            "pid": proc.pid,
            "log_path": str(log_path),
            "label": label,
            "started_at": info["started_at"],
            "next_step": (
                "Poll talon_status to see when it completes, or "
                "talon_job_log(job_id) to see live output."
            ),
        }

    async def _dispatch_via_cli(self, args: List[str]) -> Dict[str, Any]:
        """Fall back to kestrel-talon CLI via subprocess."""
        talon_bin = self._find_talon_bin()
        if not talon_bin:
            return {
                "dispatched": False,
                "error": (
                    "kestrel-talon not found. Set KESTREL_TALON_BIN, install "
                    "kestrel-talon into the kestrel-sovereign venv "
                    "(`uv sync`), or place a sibling checkout at "
                    "../kestrel-talon with its own .venv."
                ),
            }

        try:
            env = self._build_subprocess_env()
        except RuntimeError as e:
            return {"dispatched": False, "error": str(e)}

        cmd = [talon_bin] + args
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=300,
            )
            success = proc.returncode == 0
            return {
                "dispatched": success,
                "method": "cli",
                "command": " ".join(cmd),
                "returncode": proc.returncode,
                "stdout": stdout.decode(errors="replace")[-500:] if stdout else "",
                "stderr": stderr.decode(errors="replace")[-500:] if stderr else "",
            }
        except asyncio.TimeoutError:
            return {"dispatched": False, "error": "CLI command timed out (300s)"}
        except Exception as e:
            return {"dispatched": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_peers_feature(self):
        """Get the PeersFeature instance if available."""
        if hasattr(self.agent, '_features'):
            for f in self.agent._features:
                if type(f).__name__ == "PeersFeature":
                    return f
        return None

    def _discover_host_url(self) -> Optional[str]:
        """Discover the multi_agent host URL."""
        host_url = os.environ.get("KESTREL_HOST_URL")
        if host_url:
            return host_url.rstrip("/")

        from kestrel_sovereign.paths import project_dir
        for candidate in [
            Path.cwd() / "multi_agent.toml",
            project_dir() / "multi_agent.toml",
        ]:
            if candidate.exists():
                try:
                    import toml
                    data = toml.load(candidate)
                    port = data.get("host", {}).get("port", 8888)
                    return f"http://localhost:{port}"
                except Exception as e:
                    logger.debug(f"Could not read {candidate}: {e}")
        return None
