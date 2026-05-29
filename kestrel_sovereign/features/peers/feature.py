"""
Peers Feature — Inter-agent communication for multi_agent environments.

Allows agents to discover sibling agents and send messages to them
through the multi_agent host proxy. Works in both local and cloud
deployments. Two transports:

* ``ask_agent`` — synchronous Q&A via ``/api/agent/invoke``. Legacy
  surface kept until Epic #1367 reroutes it onto the A2A path as
  ``send_a2a_question``.

* ``send_a2a_task`` — asynchronous A2A task submission via
  ``/api/agent/tasks/send``. Persists to the recipient's TaskStore,
  fires the ``a2a.task_submitted`` signal so the recipient wakes,
  carries causation chain for cycle detection.

The legacy Mesh Protocol (send_mesh_message / mesh_inbox / receive_mesh_message
+ /agent/mesh endpoint + features/peers/mesh.py) was retired in #1367.
All Falconer workflow events (assign, review_needed, complete, etc.)
now go through send_a2a_task with ``metadata["skill"]`` set to the
corresponding ``workflow.*`` skill id.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool

logger = logging.getLogger(__name__)

# Timeout for inter-agent calls (seconds)
PEER_CONNECT_TIMEOUT = 5.0
PEER_READ_TIMEOUT = 300.0  # Local LLM responses (e.g. Kimi K2.5) can be very slow


def _discover_host_url() -> Optional[str]:
    """Discover the multi_agent host URL.

    Checks in order:
    1. KESTREL_HOST_URL env var (set by ProcessManager or manually)
    2. multi_agent.toml in project directory (read host port)
    3. None if not in a multi_agent environment
    """
    # Explicit env var (most reliable)
    host_url = os.environ.get("KESTREL_HOST_URL")
    if host_url:
        return host_url.rstrip("/")

    # Try reading multi_agent.toml to get host port. Resolve via the
    # paths module so pip-installed users land on their KESTREL_HOME /
    # ~/.kestrel project root rather than the package's site-packages
    # parent (which would never have a multi_agent.toml).
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


class PeersFeature(Feature):
    """Inter-agent communication — ask questions to sibling agents in the multi_agent."""

    @property
    def tool_description(self) -> str:
        return (
            "Communicate with other agents in the multi_agent — "
            "send messages to peer agents and list available peers"
        )

    @property
    def promote_tools_on_startup(self) -> bool:
        return True

    async def initialize(self):
        self._host_url = _discover_host_url()
        self._api_key = os.environ.get("KESTREL_API_KEY", "")
        self._own_name = self._get_own_name()

        if self._host_url:
            logger.info(f"PeersFeature initialized: host={self._host_url}, self={self._own_name}")
        else:
            logger.info("PeersFeature initialized but no multi_agent host found (standalone mode)")

    def _get_own_name(self) -> str:
        """Get this agent's name (from KESTREL_DB_PATH basename or agent node)."""
        # Try agent node name first
        if hasattr(self.agent, '_agent_name') and self.agent._agent_name:
            return self.agent._agent_name

        # Fall back to data dir basename
        db_path = os.environ.get("KESTREL_DB_PATH", "")
        if db_path:
            return Path(db_path).name

        return "unknown"

    def _build_headers(self) -> dict:
        """Build headers for inter-agent HTTP calls."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers

    @tool(
        name="list_peers",
        description="List all available peer agents in the multi_agent.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!peers"
    )
    async def list_peers(self) -> ToolResult:
        """
        Discover available peer agents via the multi_agent host.
        Returns their names, status, and capabilities.
        """
        if not self._host_url:
            # Honesty: standalone mode is not a failure (the listing
            # WAS performed and returned the truthful "0 peers"), but
            # the agent must speak that no host is configured rather
            # than narrate "found 0 peers" as if peers really were
            # absent. PARTIAL with the diagnostic in the caveat.
            return ToolResult.partial(
                confirmation="No peers (standalone mode)",
                error="Not running in a multi_agent environment — no host to query",
                data={"peers": [], "note": "Not running in a multi_agent environment"},
            )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self._host_url}/api/agents",
                    headers=self._build_headers(),
                    timeout=PEER_CONNECT_TIMEOUT,
                )
                resp.raise_for_status()
                agents_data = resp.json()
        except httpx.ConnectError:
            return ToolResult.failed(
                "Could not connect to multi_agent host",
                data={"peers": [], "error": "Could not connect to multi_agent host"},
            )
        except Exception as e:
            logger.error(f"Failed to list peers: {e}")
            return ToolResult.failed(
                str(e),
                data={"peers": [], "error": str(e)},
            )

        peers = []
        for agent in agents_data if isinstance(agents_data, list) else agents_data.get("agents", []):
            name = agent.get("name", agent.get("id", ""))
            if name.lower() != self._own_name.lower():
                peers.append({
                    "name": name,
                    "status": agent.get("status", "unknown"),
                    "description": agent.get("description", ""),
                })

        return ToolResult.ok(
            confirmation=f"Found {len(peers)} peer(s) (self={self._own_name})",
            data={"peers": peers, "self": self._own_name},
        )

    @tool(
        name="ask_agent",
        description="Send a message to another agent in the multi_agent and get their response. Use this to collaborate, ask questions, or delegate tasks to peer agents.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!ask"
    )
    async def ask_agent(self, agent_name: str, message: str) -> ToolResult:
        """
        Send a message to a peer agent and return their response.

        Args:
            agent_name: Name of the agent to message (e.g. "emma", "claw")
            message: The message or question to send
        """
        if not self._host_url:
            return ToolResult.failed(
                "Not running in a multi_agent environment — no host to proxy through",
                data={"response": None, "agent": agent_name},
            )

        if agent_name.lower() == self._own_name.lower():
            return ToolResult.failed(
                "Cannot send a message to yourself",
                data={"response": None, "agent": agent_name},
            )

        url = f"{self._host_url}/api/agents/{agent_name}/api/agent/invoke"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json={"input": message},
                    headers=self._build_headers(),
                    timeout=httpx.Timeout(
                        connect=PEER_CONNECT_TIMEOUT,
                        read=PEER_READ_TIMEOUT,
                        write=PEER_READ_TIMEOUT,
                        pool=PEER_CONNECT_TIMEOUT,
                    ),
                )
        except httpx.ConnectError:
            return ToolResult.failed(
                f"Could not reach agent '{agent_name}' — multi_agent host unreachable",
                data={"response": None, "agent": agent_name},
            )
        except httpx.TimeoutException:
            return ToolResult.failed(
                f"Agent '{agent_name}' took too long to respond",
                data={"response": None, "agent": agent_name},
            )
        except Exception as e:
            logger.error(f"Failed to message agent '{agent_name}': {e}")
            return ToolResult.failed(
                str(e),
                data={"response": None, "agent": agent_name},
            )

        if resp.status_code == 404:
            return ToolResult.failed(
                f"Agent '{agent_name}' not found in the multi_agent",
                data={"response": None, "agent": agent_name},
            )
        if resp.status_code == 503:
            return ToolResult.failed(
                f"Agent '{agent_name}' is offline",
                data={"response": None, "agent": agent_name},
            )

        try:
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return ToolResult.failed(
                str(e),
                data={"response": None, "agent": agent_name},
            )

        response_text = data.get("response", data.get("output", str(data)))
        return ToolResult.ok(
            confirmation=f"Got response from {agent_name}",
            data={"agent": agent_name, "response": response_text},
        )

    # ------------------------------------------------------------------
    # A2A: send a task to a peer agent (with inbound-wake semantics).
    # This is the supersedes-mesh direction (#645): peer-addressed
    # tasks land in the recipient's task_store AND trigger an
    # ``a2a.task_submitted`` signal that wakes their cognition loop,
    # so the recipient autonomously acts on the task rather than
    # waiting for a human-driven chat turn to notice it.
    # ------------------------------------------------------------------

    async def _post_a2a_task(
        self,
        recipient: str,
        message: str,
        skill_id: str = "",
        session_id: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[
        Optional[Dict[str, Any]],
        Optional[list],
        Optional[ToolResult],
    ]:
        """Shared POST helper for all three a2a verbs.

        Returns ``(task_data, chain, error_result)``. On success
        ``task_data`` is the Task envelope from the recipient (with
        ``id``, ``status``, etc.) and ``chain`` is the serialized
        causation chain we attached to outbound metadata (or None when
        no chain was active); on failure ``error_result`` is a
        populated ToolResult.failed envelope the caller returns
        directly. The chain is returned so question-supervisor wiring
        can rehydrate it into the resumption signal without a second
        ContextVar read after the spawn (#1444).

        Centralizing this means the three verbs (send_a2a_message,
        send_a2a_question, send_a2a_task) share identical wire
        semantics, causation-chain attachment, and error handling —
        the difference between them is only what the caller does with
        the result (fire-and-forget vs fire-and-resume vs return
        task_id).
        """
        from uuid import uuid4

        if not self._host_url:
            return None, None, ToolResult.failed(
                "Not running in a multi_agent environment — no host to proxy through",
                data={"sent": False, "recipient": recipient},
            )

        if recipient.lower() == self._own_name.lower():
            return None, None, ToolResult.failed(
                "Cannot send an A2A task to yourself",
                data={"sent": False, "recipient": recipient},
            )

        task_id = uuid4().hex
        sess_id = session_id or uuid4().hex
        url = f"{self._host_url}/api/agents/{recipient}/api/agent/tasks/send"
        outbound_metadata: Dict[str, Any] = {"sender": self._own_name}
        if skill_id:
            outbound_metadata["skill"] = skill_id
        if extra_metadata:
            outbound_metadata.update(extra_metadata)
        # Attach the in-flight signal-driven turn's causation chain so
        # the receiving agent's a2a.task_submitted signal carries the
        # lineage. Without this, A→B→A ping-pong loops bypass the
        # dispatcher's cycle detection (every inbound task starts
        # fresh at depth 1). Codex P1 on PR #1366.
        chain: Optional[list] = None
        chain_provider = getattr(self.agent, "_provide_causation_chain", None)
        if callable(chain_provider):
            try:
                chain = chain_provider()
            except Exception as e:
                logger.debug(
                    "Failed to read causation chain for outbound A2A task: %s",
                    e,
                )
                chain = None
            if chain:
                outbound_metadata["causation_chain"] = chain
        payload = {
            "id": task_id,
            "sessionId": sess_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message}],
            },
            "metadata": outbound_metadata,
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers=self._build_headers(),
                    timeout=httpx.Timeout(
                        connect=PEER_CONNECT_TIMEOUT,
                        read=PEER_READ_TIMEOUT,
                        write=PEER_READ_TIMEOUT,
                        pool=PEER_CONNECT_TIMEOUT,
                    ),
                )
        except httpx.ConnectError:
            return None, None, ToolResult.failed(
                f"Could not reach agent '{recipient}'",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        except httpx.TimeoutException:
            return None, None, ToolResult.failed(
                f"Agent '{recipient}' timed out",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        except Exception as e:
            logger.error(f"A2A send to '{recipient}' failed: {e}")
            return None, None, ToolResult.failed(
                str(e),
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )

        if resp.status_code == 404:
            return None, None, ToolResult.failed(
                f"Agent '{recipient}' not found or A2A endpoint missing",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        if resp.status_code == 503:
            return None, None, ToolResult.failed(
                f"Agent '{recipient}' is offline or TaskManager unavailable",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )

        try:
            resp.raise_for_status()
            task_data = resp.json()
        except Exception as e:
            return None, None, ToolResult.failed(
                str(e),
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )

        # Ensure id/sessionId always populate (older recipients might
        # echo only one or the other).
        task_data.setdefault("id", task_id)
        task_data.setdefault("sessionId", sess_id)
        return task_data, chain, None

    @tool(
        name="send_a2a_message",
        description=(
            "Send an async message to another agent — fire-and-forget, "
            "no reply expected. Persists in the recipient's TaskStore "
            "and fires the a2a.task_submitted signal so they wake and "
            "see it on their next cognition turn, but the caller does "
            "NOT track lifecycle. Use this for notifications, FYIs, "
            "status updates ('I just shipped PR 42'). For a tracked "
            "work assignment use send_a2a_task; for a synchronous "
            "Q&A use send_a2a_question."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a tell",
    )
    async def send_a2a_message(
        self,
        recipient: str,
        message: str,
        session_id: str = "",
    ) -> ToolResult:
        """
        Send an async fire-and-forget A2A message. The recipient's
        cognition loop wakes (a2a.task_submitted), they see the
        message, they decide whether to act — but the caller doesn't
        wait or track. Same wire as send_a2a_task but no skill_id is
        attached (signals "informational, not work assignment").
        """
        task_data, _chain, err = await self._post_a2a_task(
            recipient=recipient, message=message,
            skill_id="", session_id=session_id,
            extra_metadata={"a2a_verb": "message"},
        )
        if err is not None:
            return err
        return ToolResult.ok(
            confirmation=(
                f"A2A message sent to {recipient} "
                f"(task_id={task_data['id']}). Recipient has been signaled."
            ),
            data={
                "sent": True,
                "task_id": task_data["id"],
                "session_id": task_data["sessionId"],
                "recipient": recipient,
            },
        )

    @tool(
        name="send_a2a_question",
        description=(
            "Ask another agent a question. Fire-and-resume: this tool "
            "POSTs the question, spawns a background SSE subscription "
            "on the recipient's task, and returns IMMEDIATELY with "
            "``awaiting_reply=True``. Your current turn ends here. "
            "When the recipient's task reaches a terminal state, the "
            "``a2a.question_answered`` signal fires a fresh COGNITION "
            "turn on your dispatcher with the reply text inline — "
            "respond there. Do NOT block your turn waiting for the "
            "answer; the supervisor will wake you. For fire-and-forget "
            "use send_a2a_message; for tracked work you'll check on "
            "later use send_a2a_task."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a ask",
    )
    async def send_a2a_question(
        self,
        recipient: str,
        message: str,
        session_id: str = "",
        timeout_seconds: int = 300,
    ) -> ToolResult:
        """
        Submit an A2A question to a peer agent under the fire-and-resume
        contract (#1444). The POST happens synchronously (so transport
        failures are surfaced to the caller immediately), but the wait
        for the answer does NOT block the current turn — it's handled
        by a background subscription supervisor that:

        1. Records a ``pending_a2a_questions`` row keyed by ``task_id``
        2. Opens an SSE stream to the recipient's
           ``/api/agent/tasks/{task_id}/subscribe`` endpoint
        3. On terminal status frame (completed/failed/canceled),
           extracts the reply text, marks the pending row RESOLVED,
           and enqueues a local ``a2a.question_answered`` signal so a
           fresh COGNITION turn fires on the sender with the reply
           inline.
        4. Auto-reconnects with backoff (1s/2s/5s/10s capped) on
           transient httpx failures until terminal or deadline.

        Args:
            recipient: Peer agent name (e.g. "Meridian").
            message: The question / prompt.
            session_id: Optional A2A session id.
            timeout_seconds: Wall-clock cap on the supervisor. The
                hourly expiry sweep marks any WAITING row past this
                deadline as EXPIRED and fires a synthetic
                ``a2a.question_answered`` signal with
                ``state='expired'`` so the asking lineage still
                resumes cleanly. Default 300s.
        """
        task_data, chain, err = await self._post_a2a_task(
            recipient=recipient, message=message,
            skill_id="", session_id=session_id,
            extra_metadata={
                "a2a_verb": "question",
                "reply_expected": True,
            },
        )
        if err is not None:
            return err

        task_id = task_data["id"]
        sess_id = task_data["sessionId"]
        # Compute UTC deadline once — same value lands in the pending
        # row and in the supervisor's monotonic loop cap. We store ISO
        # for cross-backend portability, then convert back to monotonic
        # inside the supervisor.
        from datetime import datetime, timedelta, timezone
        deadline_utc = (
            datetime.now(timezone.utc)
            + timedelta(seconds=max(int(timeout_seconds), 1))
        )

        # Hard requirement: store + dispatcher must be wired. Skip with
        # a clear error message if the agent didn't initialize them
        # (e.g. mid-boot tool call) rather than silent fallback.
        store = getattr(self.agent, "pending_a2a_questions", None)
        if store is None:
            return ToolResult.failed(
                "send_a2a_question is unavailable: agent has no "
                "pending_a2a_questions store wired. This indicates a "
                "boot-order bug — file an issue.",
                data={
                    "sent": True,
                    "awaiting_reply": False,
                    "task_id": task_id,
                    "recipient": recipient,
                },
            )

        # Record the in-flight correlation row before spawning the
        # supervisor so a process crash between POST and supervisor
        # start is recoverable via the startup-replay sweep.
        try:
            await store.insert(
                task_id=task_id,
                recipient=recipient,
                original_question=message,
                origin_turn_id=self._safe_get_current_turn_id(),
                origin_session_id=sess_id,
                deadline=deadline_utc,
            )
        except Exception as e:
            # Codex round 3 P2d on PR #1453: without a pending row the
            # supervisor's mark_resolved would return False on the
            # terminal frame and silently drop the resumption signal as
            # a duplicate — the asking lineage would never resume even
            # though the task was sent and the receiver answered.
            # Surface this as a failure so the caller knows fire-and-
            # resume is NOT in play: the task was POSTed (receiver will
            # still act), but resumption is broken.
            logger.error(
                "Failed to record pending_a2a_question for task=%s "
                "recipient=%s: %s. Failing the tool call rather than "
                "silently losing the resumption signal.",
                task_id, recipient, e, exc_info=True,
            )
            return ToolResult.failed(
                f"Question was POSTed to {recipient} (task_id={task_id}) "
                f"but the local pending-questions store rejected the "
                f"correlation row ({type(e).__name__}: {e}). Without "
                f"that row, the a2a.question_answered signal cannot "
                f"fire — your turn will NOT be resumed when "
                f"{recipient} answers. The receiver will still process "
                f"the task; you can fetch the result manually with "
                f"get_peer_task_result.",
                data={
                    "sent": True,
                    "awaiting_reply": False,
                    "task_id": task_id,
                    "session_id": sess_id,
                    "recipient": recipient,
                    "store_error": f"{type(e).__name__}: {e}",
                },
            )

        # Spawn the supervisor as an agent-owned background task. It
        # runs the SSE loop, fires the a2a.question_answered signal on
        # terminal frame, and exits.
        self.agent._track_background_task(
            self._supervise_a2a_question(
                task_id=task_id,
                recipient=recipient,
                original_question=message,
                sess_id=sess_id,
                deadline_utc=deadline_utc,
                causation_chain=chain,
            ),
            name=f"a2a_question_supervisor:{recipient}:{task_id}",
        )

        return ToolResult.ok(
            confirmation=(
                f"Question sent to {recipient} (task_id={task_id}). "
                f"Your turn ends now — the a2a.question_answered "
                f"signal will fire a fresh cognition turn with the "
                f"reply when {recipient} reaches a terminal state "
                f"(or {timeout_seconds}s elapses)."
            ),
            data={
                "sent": True,
                "awaiting_reply": True,
                "task_id": task_id,
                "session_id": sess_id,
                "recipient": recipient,
                "expires_at": deadline_utc.isoformat(),
                "resume_via": "a2a.question_answered",
            },
        )

    @tool(
        name="get_peer_task_result",
        description=(
            "Fetch the current state + full reply text of an A2A "
            "task you previously sent to a peer agent. Use this when "
            "an `a2a.question_answered` signal arrived with "
            "`truncated=true` (the inline reply was clipped at 8 "
            "KiB) — this tool fetches the FULL untruncated body from "
            "the peer's task store. Returns the same envelope shape "
            "a local `get_task_result` would, but routed through the "
            "host proxy to the peer (#1444 truncation recovery path)."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a result",
    )
    async def get_peer_task_result(
        self,
        recipient: str,
        task_id: str,
    ) -> ToolResult:
        """Fetch a peer's task envelope and return the full reply
        text. Mirrors ``get_task_result`` but for tasks the caller
        SENT to a peer (not tasks in the caller's own store).

        Args:
            recipient: The peer agent name the task was sent to.
            task_id: The task id returned from
                ``send_a2a_question`` / ``send_a2a_task``.
        """
        if not self._host_url:
            return ToolResult.failed(
                "Not running in a multi_agent environment — no host "
                "to proxy through",
                data={"recipient": recipient, "task_id": task_id},
            )
        url = (
            f"{self._host_url}/api/agents/{recipient}"
            f"/api/agent/tasks/{task_id}"
        )
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    url,
                    headers=self._build_headers(),
                    timeout=httpx.Timeout(
                        connect=PEER_CONNECT_TIMEOUT,
                        read=PEER_CONNECT_TIMEOUT,
                        write=PEER_CONNECT_TIMEOUT,
                        pool=PEER_CONNECT_TIMEOUT,
                    ),
                )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            return ToolResult.failed(
                f"Could not reach peer '{recipient}' for task "
                f"{task_id}: {e}",
                data={"recipient": recipient, "task_id": task_id},
            )
        except Exception as e:
            return ToolResult.failed(
                f"Error fetching peer task {task_id} from "
                f"{recipient}: {e}",
                data={"recipient": recipient, "task_id": task_id},
            )

        if resp.status_code == 404:
            return ToolResult.failed(
                f"Task {task_id} not found on peer '{recipient}' "
                f"(either the peer evicted it or task_id is wrong)",
                data={"recipient": recipient, "task_id": task_id},
            )
        if resp.status_code != 200:
            return ToolResult.failed(
                f"Peer '{recipient}' returned HTTP {resp.status_code} "
                f"for task {task_id}",
                data={
                    "recipient": recipient,
                    "task_id": task_id,
                    "status_code": resp.status_code,
                },
            )
        try:
            data = resp.json()
        except ValueError as e:
            return ToolResult.failed(
                f"Peer '{recipient}' returned malformed JSON for "
                f"task {task_id}: {e}",
                data={"recipient": recipient, "task_id": task_id},
            )

        # Reuse the supervisor's dual-shape parser to extract the
        # reply text — handles both canonical A2A and kestrel's
        # flattened endpoint shape consistently with how the
        # ``a2a.question_answered`` signal got built in the first
        # place.
        raw_status = data.get("status")
        if isinstance(raw_status, dict):
            current_state = raw_status.get("state", "unknown")
        elif isinstance(raw_status, str):
            current_state = raw_status
        else:
            current_state = "unknown"
        reply_text = ""
        if isinstance(raw_status, dict):
            msg = raw_status.get("message") or {}
            for part in (msg.get("parts") or []):
                if isinstance(part, dict) and "text" in part:
                    reply_text = part["text"] or ""
                    break
        if not reply_text:
            top_msg = data.get("message")
            if isinstance(top_msg, str) and top_msg:
                reply_text = top_msg
        if not reply_text:
            for artifact in (data.get("artifacts") or []):
                if isinstance(artifact, dict):
                    for part in (artifact.get("parts") or []):
                        if isinstance(part, dict) and "text" in part:
                            reply_text = part["text"] or ""
                            break
                if reply_text:
                    break

        return ToolResult.ok(
            confirmation=(
                f"Fetched peer task {task_id[:8]} from {recipient} "
                f"(state={current_state}, {len(reply_text)} chars)"
            ),
            data={
                "recipient": recipient,
                "task_id": task_id,
                "state": current_state,
                "reply_text": reply_text,
                "artifacts": data.get("artifacts") or [],
            },
        )

    # ------------------------------------------------------------------
    # Subscription supervisor (#1444)
    #
    # When ``send_a2a_question`` POSTs an outbound task, this helper is
    # spawned as a tracked background coroutine. It opens an SSE stream
    # to ``GET /tasks/{task_id}/subscribe`` on the recipient, parses
    # ``status`` events, and when the terminal frame arrives:
    #
    #   - marks the pending row RESOLVED
    #   - extracts reply text from the terminal status.message.parts
    #   - builds an ``a2a.question_answered`` signal
    #   - enqueues it on the local dispatcher so a fresh COGNITION turn
    #     fires with the reply inline
    #
    # Reconnect: transient httpx failures back off 1s/2s/5s/10s capped,
    # restarting the stream until either (a) we see a terminal frame,
    # or (b) the wall-clock deadline passes. The hourly expiry sweep
    # is the deadline backstop — even if this supervisor goes silent
    # after a process crash, the resumption rail still fires.
    # ------------------------------------------------------------------

    def _safe_get_current_turn_id(self) -> Optional[str]:
        """Best-effort read of the in-flight turn id. The agent may not
        expose ``_get_current_turn_id`` in every embed (e.g. tests with
        a partial agent stub) — fall back to None rather than raising."""
        fn = getattr(self.agent, "_get_current_turn_id", None)
        if not callable(fn):
            return None
        try:
            return fn()
        except Exception:
            return None

    async def _supervise_a2a_question(
        self,
        *,
        task_id: str,
        recipient: str,
        original_question: str,
        sess_id: str,
        deadline_utc: Any,
        causation_chain: Optional[list],
    ) -> None:
        """Background coroutine: SSE-subscribe → fire signal on terminal.

        Runs until the recipient's task is terminal or ``deadline_utc``
        passes. Reconnects on transient httpx failure with exponential
        backoff. Errors are logged not re-raised — supervisor death
        must not surface as an unhandled task exception."""
        import asyncio
        from datetime import datetime, timezone

        subscribe_url = (
            f"{self._host_url}/api/agents/{recipient}"
            f"/api/agent/tasks/{task_id}/subscribe"
        )
        terminal_states = ("completed", "failed", "canceled")
        backoffs = [1.0, 2.0, 5.0, 10.0]
        backoff_idx = 0
        state: Optional[str] = None
        reply_text = ""

        def _remaining() -> float:
            return max(
                0.0,
                (deadline_utc - datetime.now(timezone.utc)).total_seconds(),
            )

        while _remaining() > 0 and state not in terminal_states:
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "GET",
                        subscribe_url,
                        headers=self._build_headers(),
                    ) as resp:
                        if resp.status_code == 404:
                            # Hard cut: recipient lacks the /subscribe
                            # endpoint (legacy build). Don't burn the
                            # whole deadline reconnecting.
                            logger.error(
                                "A2A question supervisor for task=%s "
                                "recipient=%s: /subscribe returned 404. "
                                "Recipient does not expose the async "
                                "question protocol. Marking the pending "
                                "row resolved with state=failed.",
                                task_id, recipient,
                            )
                            state = "failed"
                            reply_text = (
                                f"Recipient '{recipient}' does not "
                                f"expose /tasks/{{id}}/subscribe — "
                                f"upgrade them to the build that ships "
                                f"the fire-and-resume A2A question "
                                f"protocol (#1444)."
                            )
                            break
                        if resp.status_code != 200:
                            raise httpx.RequestError(
                                f"subscribe HTTP {resp.status_code}"
                            )
                        # Successful connect — reset backoff.
                        backoff_idx = 0
                        async for sse_event in self._iter_sse_events(resp):
                            # Codex round 3 P2c on PR #1453: enforce
                            # the deadline INSIDE the stream loop. On
                            # a healthy long-running task the receiver
                            # keeps the connection open emitting
                            # status/keepalive frames; without this
                            # check the supervisor blows past
                            # ``timeout_seconds`` without firing the
                            # deadline-accurate expired signal.
                            if _remaining() <= 0:
                                break
                            event_name = sse_event.get("event") or "message"
                            data_str = sse_event.get("data") or ""
                            if event_name in ("keepalive", "ping"):
                                continue
                            if event_name != "status":
                                continue
                            parsed = self._parse_sse_status_data(data_str)
                            if not parsed:
                                continue
                            event_state, event_reply = parsed
                            if event_state in terminal_states:
                                state = event_state
                                reply_text = event_reply
                                break
                        # Stream ended cleanly — if we saw a terminal,
                        # exit the outer loop; otherwise reconnect (or
                        # the outer ``while`` will exit if the deadline
                        # passed during the stream read).
                        if state in terminal_states:
                            break
            except (httpx.RequestError, httpx.TimeoutException) as e:
                logger.debug(
                    "A2A subscription stream for task=%s recipient=%s "
                    "dropped (%s); backing off",
                    task_id, recipient, e,
                )
            except Exception as e:
                logger.warning(
                    "A2A subscription supervisor for task=%s "
                    "recipient=%s unexpected error: %s",
                    task_id, recipient, e,
                )

            if state in terminal_states:
                break
            # Backoff + retry, but only if we have remaining wall-clock.
            backoff = backoffs[min(backoff_idx, len(backoffs) - 1)]
            backoff_idx += 1
            await asyncio.sleep(min(backoff, _remaining()))

        if state not in terminal_states:
            # Deadline passed without terminal. Fire the synthetic
            # ``state='expired'`` signal NOW (deadline-accurate) rather
            # than letting the caller wait up to an hour for the hourly
            # sweep — promised wake-by-deadline must actually happen at
            # the deadline (codex round 2 P2a on PR #1453). Mark-expired
            # FIRST so a racing hourly sweep that's also walking this row
            # gets a False return and drops its duplicate signal.
            logger.info(
                "A2A subscription supervisor for task=%s recipient=%s "
                "exited at deadline without terminal frame. Firing "
                "deadline-accurate expired signal.",
                task_id, recipient,
            )
            store = getattr(self.agent, "pending_a2a_questions", None)
            if store is not None:
                try:
                    was_waiting = await store.mark_expired(task_id)
                except Exception as e:
                    logger.warning(
                        "Failed to mark pending_a2a_question task=%s "
                        "expired: %s. Firing signal anyway — better a "
                        "possible duplicate than a missed resumption.",
                        task_id, e,
                    )
                    was_waiting = True
                if not was_waiting:
                    # Someone else (hourly sweep that beat us by a tick)
                    # got there first — drop our duplicate signal.
                    return
            await self._fire_question_answered_signal(
                task_id=task_id,
                recipient=recipient,
                original_question=original_question,
                sess_id=sess_id,
                state="expired",
                reply_text="",
                causation_chain=causation_chain,
            )
            return

        # Terminal: mark resolved + fire local signal. Resolve-first so
        # the startup-replay sweep doesn't double-fire if it raced this
        # supervisor to the same terminal event.
        store = getattr(self.agent, "pending_a2a_questions", None)
        if store is not None:
            try:
                was_waiting = await store.mark_resolved(task_id)
            except Exception as e:
                logger.warning(
                    "Failed to mark pending_a2a_question task=%s "
                    "resolved: %s. Firing signal anyway — the resumed "
                    "turn should not be lost to a write failure.",
                    task_id, e,
                )
                was_waiting = True
            if not was_waiting:
                # Someone else (startup-replay sweep, hourly expiry) got
                # there first. They own the signal fire; drop ours.
                logger.debug(
                    "A2A pending row for task=%s already terminal — "
                    "dropping duplicate signal from supervisor.",
                    task_id,
                )
                return

        await self._fire_question_answered_signal(
            task_id=task_id,
            recipient=recipient,
            original_question=original_question,
            sess_id=sess_id,
            state=state,
            reply_text=reply_text,
            causation_chain=causation_chain,
        )

    async def _fire_question_answered_signal(
        self,
        *,
        task_id: str,
        recipient: str,
        original_question: str,
        sess_id: str,
        state: str,
        reply_text: str,
        causation_chain: Optional[list],
    ) -> None:
        """Build and enqueue the local ``a2a.question_answered`` signal.

        Factored out so the supervisor AND the future startup-replay /
        hourly-expiry sweeps share one fire path. Errors here are
        logged not raised — losing a resumption signal is bad but
        crashing the dispatcher hop is worse."""
        from kestrel_sovereign.signals.sources.a2a_question_answered import (
            build_signal_for_question_answered,
        )

        dispatcher = getattr(self.agent, "dispatcher", None)
        if dispatcher is None:
            logger.error(
                "Cannot fire a2a.question_answered for task=%s — "
                "agent has no dispatcher.",
                task_id,
            )
            return

        try:
            target_agent = getattr(self.agent, "did", None) or self._own_name
            signal = build_signal_for_question_answered(
                task_id=task_id,
                recipient=recipient,
                original_question=original_question,
                reply_text=reply_text or "",
                state=state,
                target_agent=target_agent,
                origin_session_id=sess_id,
                causation_chain=causation_chain,
            )
            await dispatcher.enqueue_signal(signal)
        except Exception as e:
            logger.error(
                "Failed to enqueue a2a.question_answered for task=%s "
                "recipient=%s: %s",
                task_id, recipient, e,
                exc_info=True,
            )

    async def _iter_sse_events(self, response):
        """Parse Server-Sent Events from an httpx streaming response.

        Yields ``{event, data}`` dicts per SSE frame (terminated by a
        blank line). httpx exposes ``aiter_lines()`` which already
        strips trailing newlines, so we accumulate ``event:`` and
        ``data:`` field values until the blank-line separator. Comment
        lines (``:`` prefix) are dropped silently."""
        event_name = None
        data_lines: List[str] = []
        async for line in response.aiter_lines():
            if line == "":
                if event_name is not None or data_lines:
                    yield {
                        "event": event_name,
                        "data": "\n".join(data_lines),
                    }
                event_name = None
                data_lines = []
                continue
            if line.startswith(":"):
                # Comment / heartbeat — ignore.
                continue
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip())
            # Other SSE fields (id:, retry:) are not used by our
            # producer; ignore them.

    def _parse_sse_status_data(
        self, data_str: str,
    ) -> Optional[Tuple[str, str]]:
        """Parse a ``status`` SSE frame's data field.

        Returns ``(state, reply_text)`` on success or None if the
        frame is malformed / pre-terminal. Reply text is extracted
        from the same three locations the legacy polling code
        checked — ``status.message.parts``, top-level ``message``
        string, and ``artifacts[].parts`` — so the supervisor handles
        both the canonical A2A spec shape and kestrel's flattened
        endpoint shape (#1444 carries the same dual-shape logic
        forward from the legacy polling path)."""
        if not data_str:
            return None
        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None

        raw_status = data.get("status")
        if isinstance(raw_status, dict):
            current_state = raw_status.get("state")
        else:
            current_state = raw_status
        if not isinstance(current_state, str):
            return None

        reply_text = ""
        if isinstance(raw_status, dict):
            msg = raw_status.get("message") or {}
            for part in (msg.get("parts") or []):
                if isinstance(part, dict) and "text" in part:
                    reply_text = part["text"] or ""
                    break
        if not reply_text:
            top_msg = data.get("message")
            if isinstance(top_msg, str) and top_msg:
                reply_text = top_msg
        if not reply_text:
            for artifact in (data.get("artifacts") or []):
                if isinstance(artifact, dict):
                    for part in (artifact.get("parts") or []):
                        if isinstance(part, dict) and "text" in part:
                            reply_text = part["text"] or ""
                            break
                if reply_text:
                    break

        return current_state, reply_text

    # ------------------------------------------------------------------
    # Startup-replay + hourly expiry sweep (#1444 step 6)
    #
    # On boot, walk ``pending_a2a_questions WHERE status='WAITING'``:
    #   - past-deadline rows → mark EXPIRED + fire synthetic
    #     ``a2a.question_answered`` with state='expired' so the asking
    #     lineage resumes with a clean branch in the prompt template
    #   - within-deadline rows → spawn a fresh subscription supervisor
    #     so the SSE wait survives process restarts
    #
    # An hourly background task runs the same expired-row scan so rows
    # whose deadline lapses without a supervisor (e.g. supervisor
    # crashed) still get a synthetic terminal signal.
    # ------------------------------------------------------------------

    # Hourly cron interval — overridable via constructor for test
    # injection. 3600s is the Sovereign-decided default.
    EXPIRY_SWEEP_INTERVAL_SECONDS = 3600

    async def post_all_features_loaded(self, agent):
        """Run startup-replay and start the hourly expiry sweep.

        Called once after every feature has initialized — by that
        point the dispatcher and the ``pending_a2a_questions`` store
        are both wired on the agent. Skips silently when either is
        absent (non-multi-agent mode, no DB) or when no host URL is
        configured (no peers to subscribe to)."""
        store = getattr(agent, "pending_a2a_questions", None)
        if store is None:
            logger.debug(
                "Skipping a2a question startup-replay — no "
                "pending_a2a_questions store wired."
            )
            return
        if self._host_url is None:
            logger.debug(
                "Skipping a2a question startup-replay — no host URL."
            )
            return

        try:
            await self._replay_pending_a2a_questions(store)
        except Exception as e:
            logger.warning(
                "a2a question startup-replay failed: %s. The hourly "
                "sweep is still the backstop — operators can still "
                "resume in-flight questions.",
                e, exc_info=True,
            )

        # Hourly sweep as agent-owned background task. Auto-cancelled
        # on agent shutdown by ``_shutdown_background_tasks``.
        agent._track_background_task(
            self._hourly_expiry_sweep_loop(store),
            name="a2a_question_expiry_sweep",
        )

    async def _replay_pending_a2a_questions(self, store) -> None:
        """Walk every WAITING row at boot. Past-deadline rows get a
        synthetic ``state='expired'`` signal; within-deadline rows get
        a fresh subscription supervisor. The chain is NOT persisted
        across restarts — a restart erases the asking turn's
        context, so the resumed signal carries an empty chain and
        the dispatcher applies its normal depth-bounded cycle check
        from scratch."""
        from datetime import datetime, timezone

        waiting = await store.list_waiting()
        if not waiting:
            logger.debug("a2a question startup-replay: no WAITING rows.")
            return

        now = datetime.now(timezone.utc)
        replayed = 0
        expired = 0
        for row in waiting:
            try:
                deadline = datetime.fromisoformat(row.deadline)
            except (TypeError, ValueError):
                # Unparseable deadline → safer to expire than to spawn
                # a supervisor that might run forever.
                logger.warning(
                    "a2a startup-replay: unparseable deadline %r for "
                    "task=%s — treating as expired.",
                    row.deadline, row.task_id,
                )
                await self._handle_expired_row(store, row)
                expired += 1
                continue
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            if deadline <= now:
                await self._handle_expired_row(store, row)
                expired += 1
                continue
            # Within deadline — spawn supervisor. Chain is not
            # persisted across restarts; the resumed signal fires with
            # an empty chain and the dispatcher applies its normal
            # depth-bounded cycle check from scratch.
            self.agent._track_background_task(
                self._supervise_a2a_question(
                    task_id=row.task_id,
                    recipient=row.recipient,
                    original_question=row.original_question,
                    sess_id=row.origin_session_id or "",
                    deadline_utc=deadline,
                    causation_chain=None,
                ),
                name=(
                    f"a2a_question_supervisor:replay:"
                    f"{row.recipient}:{row.task_id}"
                ),
            )
            replayed += 1

        logger.info(
            "a2a question startup-replay: replayed=%d expired=%d "
            "total_waiting=%d",
            replayed, expired, len(waiting),
        )

    async def _hourly_expiry_sweep_loop(self, store) -> None:
        """Sweep ``list_waiting_past_deadline`` every hour. For each
        row mark EXPIRED + fire a synthetic ``a2a.question_answered``
        signal with ``state='expired'`` so the asking lineage
        resumes with a clean branch in the prompt template. Logs
        and continues on transient failures — this loop is the
        deadline backstop and must not die silently."""
        import asyncio

        while True:
            try:
                await asyncio.sleep(self.EXPIRY_SWEEP_INTERVAL_SECONDS)
                expired = await store.list_waiting_past_deadline()
                for row in expired:
                    try:
                        await self._handle_expired_row(store, row)
                    except Exception as e:
                        logger.warning(
                            "a2a expiry sweep: failed to expire row "
                            "task=%s: %s",
                            row.task_id, e,
                        )
                if expired:
                    logger.info(
                        "a2a expiry sweep: expired=%d", len(expired),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "a2a expiry sweep iteration failed: %s. "
                    "Continuing.", e,
                )
                # Don't tight-loop on persistent failure — back off
                # then resume the normal cadence.
                await asyncio.sleep(60)

    async def _handle_expired_row(self, store, row) -> None:
        """Mark a single WAITING row EXPIRED + fire the synthetic
        ``state='expired'`` signal. Idempotent: if the row was
        already terminal (raced the supervisor), drop silently
        instead of double-firing."""
        was_waiting = await store.mark_expired(row.task_id)
        if not was_waiting:
            return
        await self._fire_question_answered_signal(
            task_id=row.task_id,
            recipient=row.recipient,
            original_question=row.original_question,
            sess_id=row.origin_session_id or "",
            state="expired",
            reply_text="",
            causation_chain=None,
        )

    @tool(
        name="send_a2a_task",
        description=(
            "Submit a tracked A2A task to another agent. Persists in "
            "the recipient's TaskStore, fires the a2a.task_submitted "
            "signal so they wake and process it, returns the task_id "
            "for tracking. Caller can poll status via get_a2a_task "
            "(or receive the a2a.task_complete signal). Use this for "
            "delegated work you'll check on later. For an answer "
            "now use send_a2a_question; for a fire-and-forget "
            "notification use send_a2a_message."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a send",
    )
    async def send_a2a_task(
        self,
        recipient: str,
        message: str,
        skill_id: str = "",
        session_id: str = "",
    ) -> ToolResult:
        """
        Submit an A2A task to a peer agent and wake their cognition loop.

        Args:
            recipient: Peer agent name (e.g. "Meridian").
            message: The task description / prompt for the recipient.
            skill_id: Optional A2A skill id from the receiver's
                AgentCard (e.g. ``"workflow.assign"``). Defaults to
                empty — the receiver routes via their default handler.
            session_id: Optional A2A session id; auto-generated when
                empty so multiple sends are independent sessions.
        """
        task_data, _chain, err = await self._post_a2a_task(
            recipient=recipient, message=message,
            skill_id=skill_id, session_id=session_id,
            extra_metadata={"a2a_verb": "task"},
        )
        if err is not None:
            return err
        return ToolResult.ok(
            confirmation=(
                f"A2A task {task_data['id']} submitted to {recipient} "
                f"(state={(task_data.get('status') or {}).get('state','?')}). "
                f"Recipient's dispatcher has been signaled."
            ),
            data={
                "sent": True,
                "task_id": task_data["id"],
                "session_id": task_data["sessionId"],
                "state": (task_data.get("status") or {}).get("state"),
                "recipient": recipient,
            },
        )

    # Agent Mesh Protocol retired in #1367. The send_mesh_message /
    # mesh_inbox / receive_mesh_message tools and the /agent/mesh
    # endpoint were replaced by send_a2a_task above (and the wider
    # send_a2a_* family in the follow-up epic). All inter-agent
    # communication now goes through /api/agent/tasks/send so it gets
    # persistence (TaskStore), lifecycle (SUBMITTED→WORKING→COMPLETED),
    # and dispatcher-driven inbound wake (a2a.task_submitted signal).
