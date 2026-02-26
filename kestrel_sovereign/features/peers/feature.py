"""
Peers Feature — Inter-agent communication for rookery environments.

Allows agents to discover sibling agents and send messages to them
through the rookery host proxy. Works in both local and cloud deployments.

Architecture:
    Agent A → PeersFeature.ask_agent("emma", "What do you think?")
        → POST http://{host}/api/agents/emma/agent/invoke
            → Host proxy → Emma's /agent/invoke endpoint
                → Emma processes, returns response
            ← Response flows back through proxy
        ← Agent A receives Emma's answer
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

import httpx

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

logger = logging.getLogger(__name__)

# Timeout for inter-agent calls (seconds)
PEER_CONNECT_TIMEOUT = 5.0
PEER_READ_TIMEOUT = 120.0  # LLM responses can be slow


def _discover_host_url() -> Optional[str]:
    """Discover the rookery host URL.

    Checks in order:
    1. KESTREL_HOST_URL env var (set by ProcessManager or manually)
    2. rookery.toml in project directory (read host port)
    3. None if not in a rookery environment
    """
    # Explicit env var (most reliable)
    host_url = os.environ.get("KESTREL_HOST_URL")
    if host_url:
        return host_url.rstrip("/")

    # Try reading rookery.toml to get host port
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


class PeersFeature(Feature):
    """Inter-agent communication — ask questions to sibling agents in the rookery."""

    @property
    def tool_description(self) -> str:
        return (
            "Communicate with other agents in the rookery — "
            "send messages to peer agents and list available peers"
        )

    async def initialize(self):
        self._host_url = _discover_host_url()
        self._api_key = os.environ.get("KESTREL_API_KEY", "")
        self._own_name = self._get_own_name()

        if self._host_url:
            logger.info(f"PeersFeature initialized: host={self._host_url}, self={self._own_name}")
        else:
            logger.info("PeersFeature initialized but no rookery host found (standalone mode)")

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
        description="List all available peer agents in the rookery.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!peers"
    )
    async def list_peers(self) -> Dict[str, Any]:
        """
        Discover available peer agents via the rookery host.
        Returns their names, status, and capabilities.
        """
        if not self._host_url:
            return {"peers": [], "note": "Not running in a rookery environment"}

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
            return {"peers": [], "error": "Could not connect to rookery host"}
        except Exception as e:
            logger.error(f"Failed to list peers: {e}")
            return {"peers": [], "error": str(e)}

    @tool(
        name="ask_agent",
        description="Send a message to another agent in the rookery and get their response. Use this to collaborate, ask questions, or delegate tasks to peer agents.",
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
                "error": "Not running in a rookery environment — no host to proxy through",
            }

        if agent_name.lower() == self._own_name.lower():
            return {
                "response": None,
                "error": "Cannot send a message to yourself",
            }

        url = f"{self._host_url}/api/agents/{agent_name}/agent/invoke"

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
                    "error": f"Agent '{agent_name}' not found in the rookery",
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
                "error": f"Could not reach agent '{agent_name}' — rookery host unreachable",
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
