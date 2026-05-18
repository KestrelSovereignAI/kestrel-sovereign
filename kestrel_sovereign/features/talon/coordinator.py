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
from kestrel_sovereign.features.peers.mesh import (
    MeshMessage,
    MeshMessageType,
    make_assign_message,
)
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

    def __init__(self, agent):
        super().__init__(agent)
        # message_id -> {pid, started_at, log_path, command, repo,
        # issue, status, returncode, completed_at, process}
        self._jobs: Dict[str, Dict[str, Any]] = {}

    @property
    def tool_description(self) -> str:
        return "Dispatch work to Talon autonomous coding agent and monitor status"

    async def initialize(self):
        logger.info("TalonCoordinatorFeature initialized")

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

        # Mesh dispatch only carries repo/issue today, so using it for a
        # non-default runtime would silently ignore the agent's Talon controls.
        use_mesh = (
            resolved_backend == "claude"
            and resolved_model == "opus"
            and resolved_auth_lane == "oauth"
            and backend is None
            and model is None
            and auth_lane is None
        )
        if use_mesh:
            mesh_result = await self._dispatch_via_mesh(repo, issue)
            if mesh_result.get("dispatched"):
                # Mesh dispatch returns ``message_id`` (not ``job_id``) —
                # that's the tracking id the agent/user needs to follow up.
                tracking_id = (
                    mesh_result.get("job_id")
                    or mesh_result.get("message_id")
                    or "?"
                )
                return ToolResult.ok(
                    confirmation=(
                        f"Dispatched {repo}#{issue} to talon via mesh "
                        f"(message_id={tracking_id})"
                    ),
                    data=mesh_result,
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
            labels: Optional comma-separated label names.
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
            return ToolResult.failed(
                "ComputerUseFeature unavailable — refusing to file the "
                "issue via an unaudited shell. Enable computer_use so the "
                "scoped auto-approve policy and audit row apply.",
                data={"filed": False, "dispatched": False},
            )

        cmd_parts = [
            "gh", "issue", "create",
            "-R", repo_resolved,
            "--title", title,
            "--body", body,
        ]
        for lab in (
            l.strip() for l in (labels or "").split(",") if l.strip()
        ):
            cmd_parts += ["--label", lab]
        command = shlex.join(cmd_parts)

        shell_res = await cu.shell(command, timeout=120)
        shell_data = shell_res.data or {}
        stdout = str(shell_data.get("stdout", ""))
        # ToolResultStatus compares equal to its string value ("ok").
        succeeded = shell_res.status == "ok"

        # `gh issue create` prints the new issue URL on stdout.
        m = re.search(r"https://github\.com/\S+/issues/(\d+)", stdout)
        if not succeeded or m is None:
            return ToolResult.failed(
                "gh issue create did not return a parseable issue URL "
                f"(status={shell_res.status}). The command may have been "
                "denied at the approval gate, or `gh` is not authenticated "
                "in the agent's environment.",
                data={
                    "filed": False,
                    "dispatched": False,
                    "shell_status": str(shell_res.status),
                    "shell_error": shell_res.error,
                    "stdout_tail": stdout[-300:],
                },
            )

        issue_url = m.group(0)
        issue_number = int(m.group(1))

        claim = await self.talon_claim(repo=repo_resolved, issue=issue_number)
        claim_data = claim.data or {}
        dispatched = bool(claim_data.get("dispatched"))
        job_id = (
            claim_data.get("job_id")
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
            "claim": claim_data,
        }
        if dispatched:
            return ToolResult.ok(
                confirmation=(
                    f"Filed {repo_resolved}#{issue_number} ({issue_url}) "
                    f"and dispatched it to Talon (job_id={job_id})."
                ),
                data=result_data,
            )
        return ToolResult.partial(
            confirmation=(
                f"Filed {repo_resolved}#{issue_number} ({issue_url}) but "
                f"the Talon dispatch did not take."
            ),
            error=claim.error or "talon_claim did not report dispatched",
            data=result_data,
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
        # Reap finished background CLI jobs
        for jid, info in list(self._jobs.items()):
            if info.get("method") != "cli_background":
                continue
            if info.get("status") not in ("dispatched", "running"):
                continue
            proc = info.get("process")
            if proc is None:
                continue
            rc = proc.returncode
            if rc is None:
                info["status"] = "running"
                continue
            info["status"] = "complete" if rc == 0 else "failed"
            info["returncode"] = rc
            info["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Then layer on mesh inbox completions for jobs dispatched via mesh.
        # peers.mesh_inbox is async and now returns a ToolResult envelope
        # (#1061 wave 16); the legacy {"messages": [...]} dict lives under
        # .data. The pre-migration code was missing the `await` here —
        # restore correctness now that we're touching this call site.
        peers = self._get_peers_feature()
        if peers:
            inbox_envelope = await peers.mesh_inbox(limit=50)
            inbox_data = inbox_envelope.data or {}
            for msg_data in inbox_data.get("messages", []):
                msg_type = msg_data.get("type", "")
                if msg_type in ("complete", "reject"):
                    job_id = msg_data.get("correlation_id", "")
                    if job_id in self._jobs:
                        self._jobs[job_id]["status"] = msg_type
                        self._jobs[job_id]["result"] = msg_data.get("payload", {})

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
            if info.get("status") in ("complete", "failed", "reject")
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

    async def _dispatch_via_mesh(
        self, repo: str, issue_number: int, title: str = ""
    ) -> Dict[str, Any]:
        """Send an assign message to Talon via mesh protocol."""
        host_url = self._discover_host_url()
        if not host_url:
            return {"dispatched": False, "reason": "no_mesh_host"}

        msg = make_assign_message(
            sender=getattr(self.agent, 'agent_name', 'kestrel'),
            recipient="talon",
            repo=repo,
            issue_number=issue_number,
            issue_title=title or f"#{issue_number}",
        )

        url = f"{host_url}/api/agents/talon/api/agent/mesh"
        payload = json.dumps(msg.to_dict()).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = await asyncio.to_thread(
                lambda: urllib.request.urlopen(req, timeout=10).read()
            )
            self._jobs[msg.id] = {
                "repo": repo, "issue": issue_number,
                "status": "dispatched", "method": "mesh",
            }
            return {
                "dispatched": True, "method": "mesh",
                "message_id": msg.id, "repo": repo, "issue": issue_number,
            }
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            logger.debug(f"Mesh dispatch failed: {e}")
            return {"dispatched": False, "reason": "mesh_unavailable", "error": str(e)}

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
        cmd = [talon_bin] + args

        try:
            log_file = open(log_path, "w", encoding="utf-8")
        except OSError as e:
            return {"dispatched": False, "error": f"Cannot open log file: {e}"}

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
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
            "process": proc,
        }
        if extra_meta:
            info.update(extra_meta)
        self._jobs[job_id] = info

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
