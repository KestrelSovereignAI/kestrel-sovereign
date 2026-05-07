"""
Peers Feature — Inter-agent communication for multi_agent environments.

Allows agents to discover sibling agents and send messages to them
through the multi_agent host proxy. Works in both local and cloud deployments.

Includes the Agent Mesh Protocol for structured message exchange between
Falconer agents (Claws, Talon, Eye, Flight).

Architecture:
    Agent A → PeersFeature.ask_agent("emma", "What do you think?")
        → POST http://{host}/api/agents/emma/agent/invoke
            → Host proxy → Emma's /agent/invoke endpoint
                → Emma processes, returns response
            ← Response flows back through proxy
        ← Agent A receives Emma's answer

    Mesh Protocol:
        Claws → send_mesh_message(talon, assign, {issue: 42})
            → POST http://{host}/api/agents/talon/agent/mesh
            ← Talon acknowledges receipt
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory

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

    # Try reading multi_agent.toml to get host port
    for candidate in [
        Path.cwd() / "multi_agent.toml",
        Path(__file__).resolve().parents[3] / "multi_agent.toml",
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

    async def initialize(self):
        self._host_url = _discover_host_url()
        self._api_key = os.environ.get("KESTREL_API_KEY", "")
        self._own_name = self._get_own_name()
        self._mesh_inbox: List[Dict[str, Any]] = []
        self._mesh_log: List[Dict[str, Any]] = []

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
    async def list_peers(self) -> Dict[str, Any]:
        """
        Discover available peer agents via the multi_agent host.
        Returns their names, status, and capabilities.
        """
        if not self._host_url:
            return {"peers": [], "note": "Not running in a multi_agent environment"}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self._host_url}/api/agents",
                    headers=self._build_headers(),
                    timeout=PEER_CONNECT_TIMEOUT,
                )
                resp.raise_for_status()
                agents_data = resp.json()

            # Filter out self
            peers = []
            for agent in agents_data if isinstance(agents_data, list) else agents_data.get("agents", []):
                name = agent.get("name", agent.get("id", ""))
                if name.lower() != self._own_name.lower():
                    peers.append({
                        "name": name,
                        "status": agent.get("status", "unknown"),
                        "description": agent.get("description", ""),
                    })

            return {"peers": peers, "self": self._own_name}

        except httpx.ConnectError:
            return {"peers": [], "error": "Could not connect to multi_agent host"}
        except Exception as e:
            logger.error(f"Failed to list peers: {e}")
            return {"peers": [], "error": str(e)}

    @tool(
        name="ask_agent",
        description="Send a message to another agent in the multi_agent and get their response. Use this to collaborate, ask questions, or delegate tasks to peer agents.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!ask"
    )
    async def ask_agent(self, agent_name: str, message: str) -> Dict[str, Any]:
        """
        Send a message to a peer agent and return their response.

        Args:
            agent_name: Name of the agent to message (e.g. "emma", "claw")
            message: The message or question to send
        """
        if not self._host_url:
            return {
                "response": None,
                "error": "Not running in a multi_agent environment — no host to proxy through",
            }

        if agent_name.lower() == self._own_name.lower():
            return {
                "response": None,
                "error": "Cannot send a message to yourself",
            }

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

            if resp.status_code == 404:
                return {
                    "response": None,
                    "error": f"Agent '{agent_name}' not found in the multi_agent",
                }
            if resp.status_code == 503:
                return {
                    "response": None,
                    "error": f"Agent '{agent_name}' is offline",
                }

            resp.raise_for_status()
            data = resp.json()

            # Extract the response text from the agent's reply
            response_text = data.get("response", data.get("output", str(data)))

            return {
                "agent": agent_name,
                "response": response_text,
            }

        except httpx.ConnectError:
            return {
                "response": None,
                "error": f"Could not reach agent '{agent_name}' — multi_agent host unreachable",
            }
        except httpx.TimeoutException:
            return {
                "response": None,
                "error": f"Agent '{agent_name}' took too long to respond",
            }
        except Exception as e:
            logger.error(f"Failed to message agent '{agent_name}': {e}")
            return {
                "response": None,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Agent Mesh Protocol
    # ------------------------------------------------------------------

    @tool(
        name="send_mesh_message",
        description="Send a structured mesh message to a peer agent. Use for task assignments, review requests, completions, and rejections between Falconer agents.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!mesh send",
    )
    async def send_mesh_message(
        self,
        recipient: str,
        message_type: str,
        payload_json: str,
        priority: str = "normal",
        repo: str = "",
        correlation_id: str = "",
    ) -> Dict[str, Any]:
        """
        Send a structured mesh message to a peer agent.

        The message is delivered to the recipient's /agent/mesh endpoint
        via the multi_agent host proxy. The recipient stores it in their
        mesh inbox for processing.

        Args:
            recipient: Name of the target agent (e.g. "talon", "eye")
            message_type: One of: assign, review_needed, complete, reject, status_update
            payload_json: JSON string with type-specific data
            priority: Priority level: critical, high, normal, low
            repo: GitHub repo in "owner/name" format (optional)
            correlation_id: Links to a previous message (for replies)
        """
        from .mesh import MeshMessage, MeshMessageType, MeshPriority

        if not self._host_url:
            return {"sent": False, "error": "Not running in a multi_agent environment"}

        try:
            msg_type = MeshMessageType(message_type)
        except ValueError:
            valid = [t.value for t in MeshMessageType]
            return {"sent": False, "error": f"Invalid message_type. Must be one of: {valid}"}

        try:
            msg_priority = MeshPriority(priority)
        except ValueError:
            valid = [p.value for p in MeshPriority]
            return {"sent": False, "error": f"Invalid priority. Must be one of: {valid}"}

        try:
            payload = json.loads(payload_json) if isinstance(payload_json, str) else payload_json
        except json.JSONDecodeError as e:
            return {"sent": False, "error": f"Invalid payload_json: {e}"}

        msg = MeshMessage(
            type=msg_type,
            sender=self._own_name,
            recipient=recipient,
            priority=msg_priority,
            payload=payload,
            repo=repo or None,
            correlation_id=correlation_id or None,
        )

        # Deliver via multi_agent host → recipient's /api/agent/mesh endpoint
        url = f"{self._host_url}/api/agents/{recipient}/api/agent/mesh"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json=msg.to_dict(),
                    headers=self._build_headers(),
                    timeout=httpx.Timeout(
                        connect=PEER_CONNECT_TIMEOUT,
                        read=PEER_READ_TIMEOUT,
                        write=PEER_READ_TIMEOUT,
                        pool=PEER_CONNECT_TIMEOUT,
                    ),
                )

            if resp.status_code == 404:
                return {"sent": False, "error": f"Agent '{recipient}' not found or mesh endpoint not available"}
            if resp.status_code == 503:
                return {"sent": False, "error": f"Agent '{recipient}' is offline"}

            resp.raise_for_status()

            # Log the outbound message
            self._mesh_log.append(msg.to_dict())
            logger.info(f"Mesh message sent: {msg.type.value} → {recipient} (id={msg.id})")

            return {
                "sent": True,
                "message_id": msg.id,
                "type": msg.type.value,
                "recipient": recipient,
            }

        except httpx.ConnectError:
            return {"sent": False, "error": f"Could not reach agent '{recipient}'"}
        except httpx.TimeoutException:
            return {"sent": False, "error": f"Agent '{recipient}' timed out"}
        except Exception as e:
            logger.error(f"Mesh send to '{recipient}' failed: {e}")
            return {"sent": False, "error": str(e)}

    @tool(
        name="mesh_inbox",
        description="View incoming mesh messages from peer agents. Shows assignments, review requests, and status updates.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!mesh inbox",
    )
    async def mesh_inbox(self, limit: int = 20) -> Dict[str, Any]:
        """
        View recent incoming mesh messages.

        Args:
            limit: Maximum number of messages to return (default 20).
        """
        messages = self._mesh_inbox[-limit:] if limit > 0 else self._mesh_inbox
        return {
            "messages": messages,
            "count": len(messages),
            "total": len(self._mesh_inbox),
        }

    def receive_mesh_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receive and store an incoming mesh message (called by the /agent/mesh endpoint).

        Returns acknowledgement dict.
        """
        from .mesh import MeshMessage

        try:
            msg = MeshMessage.from_dict(data)
        except (KeyError, ValueError) as e:
            return {"accepted": False, "error": f"Invalid mesh message: {e}"}

        self._mesh_inbox.append(msg.to_dict())
        logger.info(f"Mesh message received: {msg.type.value} from {msg.sender} (id={msg.id})")

        return {
            "accepted": True,
            "message_id": msg.id,
            "type": msg.type.value,
        }
