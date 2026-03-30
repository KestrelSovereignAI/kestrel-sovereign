"""
In-process multi-agent manager for Kestrel.

Manages multiple KestrelAgent instances within a single process.
Used for Cloud Run deployments where running separate processes per agent
is impractical, and for any deployment that wants multi-agent in one server.

Replaces ProcessManager for in-process use; ProcessManager is still
available for local dev (separate OS processes per agent).
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.spawn.mandate import SpawnMandate, sign_mandate
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.storage.async_storage import AsyncStorage

from .config import LocalAgentConfig, RookeryConfig

logger = logging.getLogger(__name__)


async def _get_agent_did(storage_dir: str) -> str:
    """Retrieve an agent's DID from its database."""
    db_path = os.path.join(storage_dir, "kestrel_prime.db")
    storage = AsyncStorage(db_path)
    await storage.initialize()
    try:
        agent_nodes = await storage.get_nodes_by_type("agent")
        if not agent_nodes:
            raise ValueError(
                f"No agent found in {storage_dir}. "
                "Run inception first: kestrel create <name>"
            )
        return agent_nodes[0].node_id
    finally:
        await storage.close()


class AgentManager:
    """In-process multi-agent manager.

    Holds multiple KestrelAgent instances keyed by name.
    Each agent gets its own LLMService (mutable model preference state)
    and its own storage (SQLite file or Postgres with agent_id isolation).
    """

    def __init__(self, base_data_dir: Optional[Path] = None):
        self._agents: dict[str, KestrelAgent] = {}
        self._agent_names: dict[str, str] = {}  # agent_id -> name (reverse lookup)
        self._parent_children: dict[str, list[str]] = {}  # parent_did -> [child_name]
        self._child_mandates: dict[str, SpawnMandate] = {}  # child_name -> mandate
        self._base_data_dir = base_data_dir or Path.cwd()
        self._lock = asyncio.Lock()

    async def load_agent(self, name: str, config: LocalAgentConfig) -> KestrelAgent:
        """Create and initialize a KestrelAgent from a rookery config entry.

        Args:
            name: Agent name (used as routing key).
            config: LocalAgentConfig with data_dir, port, autostart.

        Returns:
            Initialized KestrelAgent instance.

        Raises:
            ValueError: If agent data directory is invalid.
        """
        resolved_dir = (self._base_data_dir / config.data_dir).resolve()

        # Validate the data directory
        errors = config.validate_runtime(base_dir=self._base_data_dir)
        if errors:
            raise ValueError(f"Agent '{name}' validation failed: {'; '.join(errors)}")

        # Get DID from the agent's database
        agent_did = await _get_agent_did(str(resolved_dir))

        # Check database backend
        db_backend = os.environ.get("KESTREL_DB_BACKEND", "sqlite")
        database_url = os.environ.get("KESTREL_DATABASE_URL")

        db_path = str(resolved_dir / "kestrel_prime.db")

        # Each agent gets its own LLMService (mutable model state)
        llm_service = LLMService()

        if db_backend.lower() == "postgres" and database_url:
            agent = KestrelAgent(
                did=agent_did,
                storage_path=db_path,
                llm_service=llm_service,
                database_url=database_url,
                db_backend="postgres",
            )
        else:
            agent = KestrelAgent(
                did=agent_did,
                storage_path=db_path,
                llm_service=llm_service,
            )

        await agent.initialize()

        self._agents[name] = agent
        self._agent_names[agent.agent_id] = name
        logger.info(f"Loaded agent '{name}' (DID: {agent_did[:30]}...)")
        return agent

    async def load_from_config(self, config: RookeryConfig) -> int:
        """Load all autostart agents from a RookeryConfig.

        Returns:
            Number of agents successfully loaded.
        """
        loaded = 0
        for name, agent_cfg in config.agents.items():
            if not isinstance(agent_cfg, LocalAgentConfig):
                logger.info(f"Skipping remote agent '{name}' (not supported in-process)")
                continue
            if not agent_cfg.autostart:
                logger.info(f"Skipping agent '{name}' (autostart=false)")
                continue
            try:
                await self.load_agent(name, agent_cfg)
                loaded += 1
            except Exception as e:
                logger.error(f"Failed to load agent '{name}': {e}")
        return loaded

    def get_agent(self, name: str) -> Optional[KestrelAgent]:
        """Get an agent by name (case-insensitive)."""
        # Try exact match first
        agent = self._agents.get(name)
        if agent:
            return agent
        # Try case-insensitive
        name_lower = name.lower()
        for key, agent in self._agents.items():
            if key.lower() == name_lower:
                return agent
        return None

    def list_agents(self) -> dict[str, KestrelAgent]:
        """Return all loaded agents as {name: agent}."""
        return dict(self._agents)

    def get_agent_name(self, agent_id: str) -> Optional[str]:
        """Get the name for an agent by its DID."""
        return self._agent_names.get(agent_id)

    async def remove_agent(self, name: str) -> bool:
        """Shutdown and remove an agent.

        Returns:
            True if agent was found and removed.
        """
        async with self._lock:
            agent = self._agents.pop(name, None)
            if not agent:
                return False
            agent_id = agent.agent_id
            self._agent_names.pop(agent_id, None)
            try:
                await asyncio.wait_for(agent.shutdown(), timeout=5.0)
                logger.info(f"Agent '{name}' shut down")
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Agent '{name}' shutdown issue: {e}")
            return True

    async def create_agent(self, name: str, parent_did: str = None) -> KestrelAgent:
        """Create a new agent via inception and load it.

        Runs the inception service to generate a new DID and database,
        then loads the agent into the manager.

        Args:
            name: Name for the new agent (used as directory name and routing key).
            parent_did: Optional DID of parent agent for delegation chain.

        Returns:
            The newly created and initialized KestrelAgent.

        Raises:
            ValueError: If an agent with this name already exists or inception fails.
        """
        if self.get_agent(name) is not None:
            raise ValueError(f"Agent '{name}' already exists")

        from kestrel_sovereign.inception_service import create_kestrel_identity_async

        agent_dir = self._base_data_dir / "agent_data" / name
        agent_dir.mkdir(parents=True, exist_ok=True)

        try:
            await create_kestrel_identity_async(
                output_dir=str(agent_dir),
                agent_name=name,
                parent_did=parent_did,
            )
        except Exception as e:
            raise ValueError(f"Inception failed for '{name}': {e}")

        config = LocalAgentConfig(
            data_dir=Path("agent_data") / name,
            port=8800 + len(self._agents) + 1,
            autostart=True,
        )
        return await self.load_agent(name, config)

    async def spawn_agent(
        self,
        name: str,
        parent_agent: KestrelAgent,
        mandate: SpawnMandate,
    ) -> KestrelAgent:
        """Create a child agent governed by a SpawnMandate.

        The child is created via inception, registered under the parent's
        DID in the parent-child tracking map, and its mandate is stored.

        Args:
            name: Name for the child agent.
            parent_agent: The parent KestrelAgent requesting the spawn.
            mandate: SpawnMandate describing purpose, budget, TTL, etc.

        Returns:
            The newly created and initialized child KestrelAgent.

        Raises:
            ValueError: If an agent with this name already exists or inception fails.
        """
        # Sign the mandate with the parent's private key if available
        parent_private_key = getattr(parent_agent, '_private_key', None)
        if parent_private_key is not None:
            sign_mandate(mandate, parent_private_key)

        # Create the child via the existing create_agent flow
        child = await self.create_agent(name, parent_did=parent_agent.agent_id)

        # Fill in child DID on the mandate
        mandate.child_did = child.agent_id

        # Track parent-child relationship
        parent_did = parent_agent.agent_id
        if parent_did not in self._parent_children:
            self._parent_children[parent_did] = []
        self._parent_children[parent_did].append(name)

        # Store the mandate
        self._child_mandates[name] = mandate

        logger.info(
            f"Spawned child '{name}' (DID: {child.agent_id[:30]}...) "
            f"for parent '{parent_did[:30]}...' — purpose: {mandate.purpose}"
        )
        return child

    def get_children(self, parent_did: str) -> list[str]:
        """Get list of child agent names for a parent DID."""
        return list(self._parent_children.get(parent_did, []))

    def get_mandate(self, child_name: str) -> Optional[SpawnMandate]:
        """Get the SpawnMandate for a child agent."""
        return self._child_mandates.get(child_name)

    async def terminate_child(self, parent_did: str, child_name: str) -> bool:
        """Terminate a specific child agent and its descendants.

        Removes the child from the parent-child map, terminates any
        grandchildren (cascading), then shuts down the child itself.

        Args:
            parent_did: DID of the parent agent.
            child_name: Name of the child to terminate.

        Returns:
            True if the child was found and terminated.
        """
        children = self._parent_children.get(parent_did, [])
        if child_name not in children:
            return False

        # Cascade: terminate grandchildren first
        child_agent = self.get_agent(child_name)
        if child_agent is not None:
            await self.terminate_children(child_agent.agent_id)

        # Remove from parent tracking
        children.remove(child_name)
        if not children:
            self._parent_children.pop(parent_did, None)

        # Clean up mandate
        self._child_mandates.pop(child_name, None)

        # Shutdown the child agent
        removed = await self.remove_agent(child_name)
        if removed:
            logger.info(f"Terminated child '{child_name}' of parent '{parent_did[:30]}...'")
        return removed

    async def terminate_children(self, parent_did: str) -> int:
        """Terminate all children of a parent agent (cascading).

        Args:
            parent_did: DID of the parent whose children to terminate.

        Returns:
            Number of children terminated.
        """
        children = list(self._parent_children.get(parent_did, []))
        count = 0
        for child_name in children:
            if await self.terminate_child(parent_did, child_name):
                count += 1
        return count

    async def shutdown_all(self) -> None:
        """Gracefully shutdown all agents."""
        # Clear parent-child tracking
        self._parent_children.clear()
        self._child_mandates.clear()

        names = list(self._agents.keys())
        for name in names:
            await self.remove_agent(name)
        logger.info("All agents shut down")
