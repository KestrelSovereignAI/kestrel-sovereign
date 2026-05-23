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
    ) -> Tuple[Optional[Dict[str, Any]], Optional[ToolResult]]:
        """Shared POST helper for all three a2a verbs.

        Returns ``(task_data, error_result)``. Exactly one is non-None:
        on success, ``task_data`` is the Task envelope from the
        recipient (with ``id``, ``status``, etc.); on failure
        ``error_result`` is a populated ToolResult.failed envelope the
        caller returns directly.

        Centralizing this means the three verbs (send_a2a_message,
        send_a2a_question, send_a2a_task) share identical wire
        semantics, causation-chain attachment, and error handling —
        the difference between them is only what the caller does with
        the result (fire-and-forget vs sync-wait vs return task_id).
        """
        from uuid import uuid4

        if not self._host_url:
            return None, ToolResult.failed(
                "Not running in a multi_agent environment — no host to proxy through",
                data={"sent": False, "recipient": recipient},
            )

        if recipient.lower() == self._own_name.lower():
            return None, ToolResult.failed(
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
            return None, ToolResult.failed(
                f"Could not reach agent '{recipient}'",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        except httpx.TimeoutException:
            return None, ToolResult.failed(
                f"Agent '{recipient}' timed out",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        except Exception as e:
            logger.error(f"A2A send to '{recipient}' failed: {e}")
            return None, ToolResult.failed(
                str(e),
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )

        if resp.status_code == 404:
            return None, ToolResult.failed(
                f"Agent '{recipient}' not found or A2A endpoint missing",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        if resp.status_code == 503:
            return None, ToolResult.failed(
                f"Agent '{recipient}' is offline or TaskManager unavailable",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )

        try:
            resp.raise_for_status()
            task_data = resp.json()
        except Exception as e:
            return None, ToolResult.failed(
                str(e),
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )

        # Ensure id/sessionId always populate (older recipients might
        # echo only one or the other).
        task_data.setdefault("id", task_id)
        task_data.setdefault("sessionId", sess_id)
        return task_data, None

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
        task_data, err = await self._post_a2a_task(
            recipient=recipient, message=message,
            skill_id="", session_id=session_id,
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
            "Ask another agent a question synchronously and return "
            "their answer. Wraps an A2A task and waits for it to "
            "reach a terminal state (COMPLETED, FAILED, or CANCELED), "
            "polling the recipient's task endpoint. Use this when you "
            "need the answer right now to continue your own turn. For "
            "fire-and-forget use send_a2a_message; for delegating "
            "tracked work you'll check on later use send_a2a_task. "
            "Note: this is the proper agent-to-agent path. ask_agent "
            "(legacy) goes through the sovereign chat endpoint, which "
            "doesn't carry sender attribution — prefer this verb."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a ask",
    )
    async def send_a2a_question(
        self,
        recipient: str,
        message: str,
        session_id: str = "",
        timeout_seconds: int = 60,
    ) -> ToolResult:
        """
        Submit an A2A task and wait synchronously for it to terminate,
        returning the recipient's answer.

        Polls ``GET /api/agents/{recipient}/api/agent/tasks/{task_id}``
        at ~1s intervals (longer as the wait extends) until the task
        reaches a terminal state or ``timeout_seconds`` elapses.

        Args:
            recipient: Peer agent name (e.g. "Meridian").
            message: The question / prompt.
            session_id: Optional A2A session id.
            timeout_seconds: Maximum wait. Default 60s; long-running
                analyses should use send_a2a_task and check back later.
        """
        import asyncio

        task_data, err = await self._post_a2a_task(
            recipient=recipient, message=message,
            skill_id="", session_id=session_id,
            extra_metadata={"reply_expected": True},
        )
        if err is not None:
            return err

        task_id = task_data["id"]
        sess_id = task_data["sessionId"]
        get_url = (
            f"{self._host_url}/api/agents/{recipient}/api/agent/tasks/{task_id}"
        )
        terminal_states = ("completed", "failed", "canceled")
        # Polling cadence: tight at first (responsive for quick
        # answers), then back off to reduce load on long waits.
        poll_intervals = [0.5, 0.5, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0]
        elapsed = 0.0

        while elapsed < timeout_seconds:
            interval = poll_intervals[min(
                int(elapsed / 5), len(poll_intervals) - 1
            )]
            await asyncio.sleep(interval)
            elapsed += interval

            try:
                async with httpx.AsyncClient() as client:
                    poll_resp = await client.get(
                        get_url,
                        headers=self._build_headers(),
                        timeout=httpx.Timeout(
                            connect=PEER_CONNECT_TIMEOUT,
                            read=PEER_CONNECT_TIMEOUT,
                            write=PEER_CONNECT_TIMEOUT,
                            pool=PEER_CONNECT_TIMEOUT,
                        ),
                    )
            except (httpx.ConnectError, httpx.TimeoutException):
                # Transient — keep waiting until the outer timeout.
                continue
            except Exception as e:
                logger.warning(
                    "A2A poll for %s/%s failed: %s",
                    recipient, task_id, e,
                )
                continue

            if poll_resp.status_code != 200:
                continue
            try:
                state_data = poll_resp.json()
            except ValueError:
                continue
            current_state = (
                (state_data.get("status") or {}).get("state")
                or state_data.get("status")
            )
            if current_state not in terminal_states:
                continue

            # Terminal state reached — extract the answer from either
            # ``message.parts[].text`` on the status (the canonical
            # A2A spot for the agent's final reply), or from any
            # ``artifacts``.
            answer_text = ""
            status = state_data.get("status") or {}
            msg = status.get("message") or {}
            for part in (msg.get("parts") or []):
                if isinstance(part, dict) and "text" in part:
                    answer_text = part["text"]
                    break
            if not answer_text:
                for artifact in (state_data.get("artifacts") or []):
                    if isinstance(artifact, dict):
                        for part in (artifact.get("parts") or []):
                            if isinstance(part, dict) and "text" in part:
                                answer_text = part["text"]
                                break
                    if answer_text:
                        break

            if current_state == "completed":
                return ToolResult.ok(
                    confirmation=(
                        f"Got answer from {recipient} "
                        f"(task_id={task_id})"
                    ),
                    data={
                        "answered": True,
                        "task_id": task_id,
                        "session_id": sess_id,
                        "recipient": recipient,
                        "answer": answer_text,
                        "state": current_state,
                    },
                )
            return ToolResult.failed(
                f"Agent '{recipient}' returned terminal state "
                f"{current_state!r}: {answer_text or '(no message)'}",
                data={
                    "answered": False,
                    "task_id": task_id,
                    "session_id": sess_id,
                    "recipient": recipient,
                    "answer": answer_text,
                    "state": current_state,
                },
            )

        # Timeout — the task may still complete later; the caller can
        # poll the task_id manually via the receiver's /tasks/{id}
        # endpoint, or switch to send_a2a_task for tracked async work.
        return ToolResult.partial(
            confirmation=(
                f"A2A question to {recipient} (task_id={task_id}) did "
                f"not reach a terminal state within "
                f"{timeout_seconds}s. The task is still live; switch "
                f"to send_a2a_task for tracked async work, or retry "
                f"with a larger timeout_seconds."
            ),
            error="timeout",
            data={
                "answered": False,
                "task_id": task_id,
                "session_id": sess_id,
                "recipient": recipient,
                "state": "timeout",
            },
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
        task_data, err = await self._post_a2a_task(
            recipient=recipient, message=message,
            skill_id=skill_id, session_id=session_id,
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
