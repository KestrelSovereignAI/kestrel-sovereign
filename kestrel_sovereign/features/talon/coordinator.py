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
import urllib.error
import urllib.request
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


class TalonCoordinatorFeature(Feature):
    """Thin dispatcher to the external kestrel-talon daemon.

    Provides !talon commands for claiming issues, batch processing,
    and checking status. Prefers mesh dispatch over CLI fallback.
    """

    def __init__(self, agent):
        super().__init__(agent)
        self._jobs: Dict[str, Dict[str, Any]] = {}  # message_id -> job info

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
        description="Dispatch a single issue to Talon for autonomous implementation.",
        category=ToolCategory.UTILITY,
        command_prefix="!talon claim",
    )
    async def talon_claim(
        self,
        repo: str,
        issue: int,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """Claim an issue for Talon to implement.

        Args:
            repo: GitHub repo in owner/name format (e.g. KestrelSovereignAI/kestrel-sovereign).
            issue: Issue number to claim.
            max_iterations: Max implementation iterations (default 3).
        """
        # Try mesh dispatch first
        result = await self._dispatch_via_mesh(repo, issue)
        if result.get("dispatched"):
            return result

        # Fallback to CLI
        return await self._dispatch_via_cli(
            ["claim", "--repo", repo, "--issue", str(issue),
             "--max-iterations", str(max_iterations)]
        )

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
            return await self._dispatch_via_cli(
                ["batch", "--prd", prd]
            )
        if label:
            return await self._dispatch_via_cli(
                ["batch", "--repo", repo, "--label", label]
            )
        return {"dispatched": False, "error": "Provide either label or prd"}

    @tool(
        name="talon_status",
        description="Check status of Talon jobs (running, completed, failed).",
        category=ToolCategory.UTILITY,
        command_prefix="!talon status",
    )
    async def talon_status(self) -> Dict[str, Any]:
        """Check status of dispatched Talon jobs."""
        # Check mesh inbox for COMPLETE/REJECT messages
        peers = self._get_peers_feature()
        completed = []
        if peers:
            inbox = peers.mesh_inbox(limit=50)
            for msg_data in inbox.get("messages", []):
                msg_type = msg_data.get("type", "")
                if msg_type in ("complete", "reject"):
                    job_id = msg_data.get("correlation_id", "")
                    if job_id in self._jobs:
                        self._jobs[job_id]["status"] = msg_type
                        self._jobs[job_id]["result"] = msg_data.get("payload", {})
                        completed.append(job_id)

        running = [
            {**info, "id": jid}
            for jid, info in self._jobs.items()
            if info.get("status") == "dispatched"
        ]
        done = [
            {**info, "id": jid}
            for jid, info in self._jobs.items()
            if info.get("status") in ("complete", "reject")
        ]

        return {
            "running": len(running),
            "completed": len(done),
            "jobs": running + done,
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
