"""TalonCoordinatorFeature - lightweight dispatch to external Talon daemon.

Wraps coordination, not the engine. Dispatches work to the external
kestrel-talon daemon via Agent Mesh Protocol (preferred) or CLI fallback.

Reference: sovereign #301
"""

import asyncio
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.peers.mesh import (
    MeshMessage,
    MeshMessageType,
    make_assign_message,
)
from kestrel_sovereign.tools.base import ToolCategory

logger = logging.getLogger(__name__)


# Default sibling-checkout layout assumed throughout: kestrel-sovereign
# and target repos live as siblings under a common project parent.
_DEFAULT_PROJECT_PARENT = Path(__file__).resolve().parents[4]


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
        max_iterations: int = 3,
        model: str = "opus",
        skip_clarification: bool = True,
        worktree: bool = True,
    ) -> Dict[str, Any]:
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
            max_iterations: Max LLM implementation iterations
                (default 3 — README recommends 2+ for non-trivial work).
            model: Claude model: ``opus``, ``sonnet``, or ``haiku``.
                Default is ``opus`` per ``feedback_kestrel_talon.md``
                — Sonnet has a track record of reading files and
                stopping without committing.
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
        # Mesh dispatch is preserved as a fast-path for environments
        # where Talon IS a registered rookery agent. In the standard
        # standalone-CLI layout it's not, and this returns dispatched
        # False quickly so we fall through to the CLI path.
        mesh_result = await self._dispatch_via_mesh(repo, issue)
        if mesh_result.get("dispatched"):
            return mesh_result

        repo_resolved = self._resolve_repo(repo)
        repo_dir = self._resolve_repo_dir(repo_resolved)
        worktree_base = (
            os.environ.get("KESTREL_TALON_WORKTREE_BASE")
            or str(_DEFAULT_PROJECT_PARENT)
        )

        args = [
            "claim",
            "--repo", repo_resolved,
            "--issue", str(issue),
            "--max-iterations", str(max_iterations),
            "--model", model,
            "--repo-dir", repo_dir,
        ]
        if worktree:
            args += ["--worktree", "--worktree-base", worktree_base]
        if skip_clarification:
            args.append("--skip-clarification")

        return await self._dispatch_via_cli_background(
            args,
            label=f"claim:{repo_resolved}#{issue}",
            extra_meta={"repo": repo_resolved, "issue": issue, "model": model},
        )

    @staticmethod
    def _resolve_repo(repo: str) -> str:
        if repo and repo.lower() == "self":
            return os.environ.get(
                "GITHUB_SELF_REPO", "KestrelSovereignAI/kestrel-sovereign"
            )
        return repo

    @staticmethod
    def _resolve_repo_dir(repo: str) -> str:
        """Find a local checkout for ``repo`` next to kestrel-sovereign.

        Talon needs the target repo on disk for git operations.
        Convention: ``<project_parent>/<repo_name>``. If that doesn't
        exist, fall back to the kestrel-sovereign project root —
        Talon will fail loudly enough that the user can supply
        ``KESTREL_TALON_REPO_DIR`` themselves rather than us
        silently dispatching against the wrong repo.
        """
        override = os.environ.get("KESTREL_TALON_REPO_DIR")
        if override:
            return override
        repo_name = repo.split("/", 1)[-1] if "/" in repo else repo
        candidate = _DEFAULT_PROJECT_PARENT / repo_name
        if candidate.is_dir():
            return str(candidate)
        # Fallback to the kestrel-sovereign root (most common case
        # since that's the agent's own repo and the most-claimed one).
        return str(_DEFAULT_PROJECT_PARENT / "kestrel-sovereign")

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
    ) -> Dict[str, Any]:
        """Dispatch batch processing to Talon.

        Args:
            repo: GitHub repo in owner/name format.
            label: Filter issues by this label.
            prd: Path to a PRD JSON file for batch mode.
        """
        if prd:
            return await self._dispatch_via_cli_background(
                ["batch", "--prd", prd],
                label=f"batch:prd={prd}",
                extra_meta={"prd": prd},
            )
        if label:
            repo_resolved = self._resolve_repo(repo)
            return await self._dispatch_via_cli_background(
                ["batch", "--repo", repo_resolved, "--label", label],
                label=f"batch:{repo_resolved}:label={label}",
                extra_meta={"repo": repo_resolved, "github_label": label},
            )
        return {"dispatched": False, "error": "Provide either label or prd"}

    @tool(
        name="talon_status",
        description="Check status of Talon jobs (running, completed, failed).",
        category=ToolCategory.UTILITY,
        command_prefix="!talon status",
    )
    async def talon_status(self) -> Dict[str, Any]:
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

        # Then layer on mesh inbox completions for jobs dispatched via mesh
        peers = self._get_peers_feature()
        if peers:
            inbox = peers.mesh_inbox(limit=50)
            for msg_data in inbox.get("messages", []):
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

        return {
            "running": len(running),
            "completed": len(done),
            "jobs": running + done,
        }

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
    ) -> Dict[str, Any]:
        """Return the last ``lines`` lines of a job's combined log."""
        info = self._jobs.get(job_id)
        if not info:
            return {"success": False, "error": f"Unknown job_id: {job_id}"}

        log_path = info.get("log_path")
        if not log_path or not os.path.isfile(log_path):
            return {
                "success": False,
                "error": f"Log file missing: {log_path}",
                "job_id": job_id,
            }

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-max(1, int(lines)):]
        except OSError as e:
            return {"success": False, "error": str(e), "job_id": job_id}

        return {
            "success": True,
            "job_id": job_id,
            "log_path": log_path,
            "status": info.get("status"),
            "returncode": info.get("returncode"),
            "lines": len(tail),
            "content": "".join(tail),
        }

    @tool(
        name="talon_pause",
        description="Pause the autonomous Talon loop (kill switch).",
        category=ToolCategory.SYSTEM,
        command_prefix="!talon pause",
    )
    async def talon_pause(self) -> Dict[str, Any]:
        """Pause the Talon dispatch loop.

        Disables the signal_dispatch scheduler so no new work is
        dispatched until resumed with !talon resume.
        """
        scheduler = getattr(self.agent, '_scheduler', None)
        if scheduler and hasattr(scheduler, 'remove_schedule'):
            scheduler.remove_schedule("signal_dispatch")
            return {"paused": True, "message": "Talon dispatch paused. Use !talon resume to restart."}
        return {"paused": False, "error": "Scheduler not available"}

    @tool(
        name="talon_resume",
        description="Resume the autonomous Talon loop after pause.",
        category=ToolCategory.SYSTEM,
        command_prefix="!talon resume",
    )
    async def talon_resume(self) -> Dict[str, Any]:
        """Resume the Talon dispatch loop after a pause."""
        scheduler = getattr(self.agent, '_scheduler', None)
        if scheduler and hasattr(scheduler, 'add_schedule'):
            scheduler.add_schedule("signal_dispatch", "5 8 * * *", "signal_dispatch")
            return {"resumed": True, "message": "Talon dispatch resumed at 08:05 daily."}
        return {"resumed": False, "error": "Scheduler not available"}

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
    async def talon_health(self) -> Dict[str, Any]:
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
            return report
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
            return report

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
            return report
        except Exception as e:
            report["execute"] = {"ok": False, "error": str(e)}
            return report

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
        return report

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
    def _build_subprocess_env() -> Dict[str, str]:
        """Construct the env dict for a kestrel-talon subprocess.

        Removes Anthropic API-key vars so talon falls back to Claude
        Max OAuth, and verifies a GitHub token is present (talon
        cannot do anything useful without one). Raises ``RuntimeError``
        with an actionable message if a required var is missing —
        callers convert that into a structured ``dispatched=False``
        response so the agent can surface the actual cause.
        """
        env = {**os.environ}
        for key in TalonCoordinatorFeature._ANTHROPIC_KEYS_TO_STRIP:
            env.pop(key, None)

        gh_token = (
            env.get("GH_TOKEN")
            or env.get("GITHUB_TOKEN")
            or env.get("GITHUB_PAT")
        )
        if not gh_token:
            raise RuntimeError(
                "kestrel-talon needs GITHUB_TOKEN, GH_TOKEN, or GITHUB_PAT "
                "in the kestrel-sovereign environment to access GitHub. "
                "Set one in .env (use `gh auth token --user UncleSaurus`)."
            )
        # Mirror to both names talon's downstream tools accept.
        env.setdefault("GITHUB_TOKEN", gh_token)
        env.setdefault("GH_TOKEN", gh_token)
        return env

    def _job_log_dir(self) -> Path:
        """Where job log files live.

        Per-agent under the agent's data directory when available
        (``<storage_path>/talon_jobs/``) so logs survive process
        restarts and aren't shared across agents in the rookery.
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
        """Discover the rookery host URL."""
        host_url = os.environ.get("KESTREL_HOST_URL")
        if host_url:
            return host_url.rstrip("/")

        for candidate in [
            Path.cwd() / "rookery.toml",
            Path(__file__).resolve().parents[3] / "rookery.toml",
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
