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

        cmd = [talon_bin] + args
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ},
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
