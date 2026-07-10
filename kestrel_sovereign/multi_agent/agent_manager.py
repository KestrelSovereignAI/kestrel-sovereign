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
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.spawn.delegated_wallet import (
    _default_currency_for,
    create_delegated_wallet,
    release_delegated_wallet,
)
from kestrel_sovereign.spawn.mandate import SpawnMandate, sign_mandate
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.storage.async_storage import AsyncStorage

from .config import LocalAgentConfig, MultiAgentConfig

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
        # child_name -> (DelegatedWallet, parent_wallet) for budgeted children, so
        # termination can release the unspent hold back to the parent (#2113).
        self._child_budgets: dict[str, tuple] = {}
        self._base_data_dir = base_data_dir or Path.cwd()
        self._lock = asyncio.Lock()
        # MONOTONIC port allocator (#1729). ``8800 + len(self._agents) + 1``
        # collides after an agent is unloaded — len shrinks and the next spawn
        # reuses a live port. A counter that only ever increases avoids reuse.
        self._port_seq = 8800
        # Hard cap on dynamically-spawned agents so a runaway spawn loop can't
        # exhaust ports / resources (#1729).
        self._max_spawned_agents = int(os.environ.get("KESTREL_MAX_SPAWNED_AGENTS", "64"))
        # In-flight spawns whose mandate isn't registered yet (counts toward the
        # cap under the lock so concurrent spawns can't race past it).
        self._pending_spawns = 0
        # Per-agent initialization failures recorded by load_from_config so
        # the FastAPI lifespan can surface them via /health (#377 lifecycle
        # hardening for multi-agent boot).
        self._init_failures: list[tuple[str, Exception]] = []

    async def load_agent(self, name: str, config: LocalAgentConfig) -> KestrelAgent:
        """Create and initialize a KestrelAgent from a multi_agent config entry.

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

        # Build allowed_features set from config (None = load all)
        allowed_features = set(config.features) if config.features is not None else None

        if db_backend.lower() == "postgres" and database_url:
            agent = KestrelAgent(
                did=agent_did,
                storage_path=db_path,
                llm_service=llm_service,
                database_url=database_url,
                db_backend="postgres",
                allowed_features=allowed_features,
            )
        else:
            agent = KestrelAgent(
                did=agent_did,
                storage_path=db_path,
                llm_service=llm_service,
                allowed_features=allowed_features,
            )

        await agent.initialize()
        # Spawn-mandate enforcement (restricted_tools hook + spawn_mandate attach)
        # is reattached inside KestrelAgent.initialize() from the persisted
        # delegation edge (#2137), so it covers every boot path — not just this
        # one — uniformly.

        self._agents[name] = agent
        self._agent_names[agent.agent_id] = name
        # Fleet-idleness (#F235): give EVERY agent — including ones created or
        # spawned after startup — a live view of all co-hosted agents, so
        # RestartCoordinator can gate a whole-host restart on the whole fleet
        # being idle. Installed here at the single registration point so a
        # dynamically-added agent can never bypass the gate. Resolves live, so
        # each agent sees agents registered after it.
        agent._cohosted_agents_provider = lambda: list(self._agents.values())
        logger.info(f"Loaded agent '{name}' (DID: {agent_did[:30]}...)")
        return agent

    async def load_from_config(self, config: MultiAgentConfig) -> int:
        """Load all autostart agents from a MultiAgentConfig.

        Per-agent failures are recorded in ``self._init_failures`` so the
        FastAPI lifespan handler can surface them via ``/health`` (lifecycle
        hardening #377 — without this, a multi-agent host whose providers
        all failed to initialize would silently report a healthy startup).

        Returns:
            Number of agents successfully loaded.
        """
        loaded = 0
        # Reset failure list — fresh load attempt.
        self._init_failures = []
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
                logger.error(f"Failed to load agent '{name}': {e}", exc_info=True)
                self._init_failures.append((name, e))
        return loaded

    @property
    def init_failures(self) -> list[tuple[str, Exception]]:
        """Read-only view of per-agent initialization failures from the last
        ``load_from_config`` call. Used by the FastAPI lifespan to surface
        lifecycle errors (e.g. ``NoLLMProvidersError``) via ``/health``.
        """
        return list(self._init_failures)

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
            if agent is not None:
                self._agent_names.pop(agent.agent_id, None)
                try:
                    await asyncio.wait_for(agent.shutdown(), timeout=5.0)
                    logger.info(f"Agent '{name}' shut down")
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"Agent '{name}' shutdown issue: {e}")

        # Release THIS agent's own budget hold AFTER it is stopped (#2113):
        # releasing before shutdown would let a still-running child spend
        # already-refunded funds. remove_agent is a SINGLE-AGENT primitive — it
        # does not cascade. Budgeted subtrees are torn down via terminate_child
        # (cascade) or shutdown_all (leaf-first), which release nested holds in
        # the correct order; directly remove_agent-ing a budgeted PARENT is not a
        # supported budget teardown (folded into #2348 with reload durability).
        # Idempotent — a no-op when those paths already released this entry.
        await self._release_child_budget(name)
        return agent is not None

    async def create_agent(
        self,
        name: str,
        parent_did: str = None,
        features: Optional[List[str]] = None,
        mandate: Optional[SpawnMandate] = None,
    ) -> KestrelAgent:
        """Create a new agent via inception and load it.

        Runs the inception service to generate a new DID and database,
        then loads the agent into the manager.

        Args:
            name: Name for the new agent (used as directory name and routing key).
            parent_did: Optional DID of parent agent for delegation chain.
            features: Optional allowlist of feature class names the agent may
                load. ``None`` loads all discovered features (backward
                compatible); a list restricts loading to those class names
                (mandatory features are always loaded regardless). Threaded
                into the agent's ``LocalAgentConfig`` so the restriction
                actually reaches ``load_agent`` / ``discover_features`` (#1946).
            mandate: Optional SpawnMandate authorizing a spawned child. When
                present it is forwarded to inception so the delegation edge
                records the mandate (purpose/ttl/max_child_depth) — without this
                the mandate never reaches ``create_kestrel_identity_async`` and
                the child is created as if unconstrained (F277).

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
                spawn_mandate=mandate,
            )
        except Exception as e:
            raise ValueError(f"Inception failed for '{name}': {e}")

        self._port_seq += 1
        config = LocalAgentConfig(
            data_dir=Path("agent_data") / name,
            port=self._port_seq,  # monotonic — never reuses an unloaded agent's port
            autostart=True,
            features=features,
        )
        return await self.load_agent(name, config)

    def _parent_feature_names(self, parent_agent: KestrelAgent) -> set[str]:
        """Feature class names available to the parent — the ceiling a child's
        ``features_allowed`` may not exceed."""
        features = getattr(parent_agent, "features", None) or {}
        try:
            return {
                type(feat).__name__ if not isinstance(key, str) else key
                for key, feat in features.items()
            }
        except AttributeError:
            return set()

    def _validate_mandate_subset(
        self, parent_agent: KestrelAgent, mandate: SpawnMandate
    ) -> None:
        """Refuse a mandate that grants the child MORE than the parent (F277).

        Uses the shared ``ScopedConstitution`` narrowing rules so this consumer
        cannot drift from the spawn-constraint contract: a child's
        ``features_allowed`` must be a subset of the parent's features, and its
        ``additional_constraints`` must be restrictions (never
        ``grant_features`` / ``override_constitution`` / ``remove_restrictions``).
        """
        from kestrel_sovereign.spawn.scoped_constitution import ScopedConstitution

        scoped = ScopedConstitution(
            base_constitution="",
            additional_constraints=getattr(mandate, "additional_constraints", {}) or {},
            features_allowed=list(getattr(mandate, "features_allowed", []) or []),
            parent_features=self._parent_feature_names(parent_agent),
        )
        ok, msg = scoped.validate_constraints()
        if not ok:
            raise ValueError(f"Spawn refused: {msg}")

    # ------------------------------------------------------------------
    # Per-child spawn budgets (#2113): hold from the parent on spawn, route the
    # child's spend through a ceiling'd DelegatedWallet, release the unspent hold
    # on termination.
    # ------------------------------------------------------------------

    @staticmethod
    def _mandate_budget(mandate: SpawnMandate) -> Decimal:
        """The mandate's requested budget as a Decimal (0 when unset/invalid)."""
        raw = getattr(mandate, "budget_allocation", 0) or 0
        try:
            return Decimal(str(raw))
        except Exception:  # noqa: BLE001 — a malformed budget is treated as none
            return Decimal("0")

    def _validate_budget_precondition(
        self, parent_agent: KestrelAgent, mandate: SpawnMandate
    ) -> None:
        """Refuse a positive budget the parent can't back (#2113), before spawn."""
        budget = self._mandate_budget(mandate)
        if budget <= 0:
            return
        # Budgets are enforced IN-PROCESS only: the ceiling + hold live in memory
        # and are released on termination/shutdown. A persistent (non-TTL) child
        # could outlive the process and be reloaded WITHOUT the delegated wrapper,
        # bypassing the cap — so restrict budgets to ephemeral (TTL-bounded)
        # children, which are torn down within the process and never reloaded.
        # Durable budgets for persistent children (persist `spent` + rehydrate on
        # load + crash reconciliation) are tracked in #2348.
        ttl = getattr(mandate, "ttl_seconds", 0) or 0
        if ttl <= 0:
            raise ValueError(
                "Spawn refused: a per-child budget requires an ephemeral child "
                "(ttl_seconds > 0). Budgets are enforced in-process and are not yet "
                "durable across a reload of a persistent child (#2348). Set a TTL, "
                "or spawn without a budget."
            )
        parent_wallet = getattr(parent_agent, "wallet", None)
        if parent_wallet is None:
            raise ValueError(
                "Spawn refused: a per-child budget requires the parent to have a "
                "funded wallet (enable the wallet feature). Spawn without a budget "
                "or fund the parent's wallet."
            )
        currency = _default_currency_for(parent_wallet)
        if not parent_wallet.can_afford(budget, currency):
            raise ValueError(
                f"Spawn refused: parent wallet cannot afford the requested budget "
                f"of {budget}."
            )

    async def _apply_delegated_budget(
        self,
        name: str,
        parent_agent: KestrelAgent,
        child: KestrelAgent,
        mandate: SpawnMandate,
    ) -> None:
        """Hold the budget from the parent and point the child's wallet at a
        ceiling'd DelegatedWallet (#2113). No-op for budget<=0."""
        budget = self._mandate_budget(mandate)
        if budget <= 0:
            return
        parent_wallet = getattr(parent_agent, "wallet", None)
        if parent_wallet is None:
            return  # precondition already refused this; defensive.
        try:
            delegated = await create_delegated_wallet(
                parent_wallet=parent_wallet,
                parent_did=parent_agent.agent_id,
                child_did=child.agent_id,
                budget=budget,
            )
        except Exception:
            # The hold failed AFTER the child was created (e.g. a concurrent
            # spend drained the parent). Don't leave an uncapped child running.
            await self.remove_agent(name)
            raise
        child.wallet = delegated
        child.wallet_agent = delegated
        # Also expose it as ``_delegated_wallet`` so the spawn-status endpoint
        # reports live budget_spent / budget_remaining (#2113).
        child._delegated_wallet = delegated
        self._child_budgets[name] = (delegated, parent_wallet)
        logger.info(
            "Applied delegated budget %s to child '%s' — spend now ceiling'd (#2113).",
            budget, name,
        )

    async def _release_child_budget(self, child_name: str) -> None:
        """Credit a terminated child's unspent budget back to its parent (#2113).

        Best-effort: a wallet error must not block termination/cleanup. The
        cascade case (a budgeted child with budgeted descendants) is handled by
        ``terminate_child`` recursing — each descendant releases its own hold.
        """
        entry = self._child_budgets.pop(child_name, None)
        if entry is None:
            return
        delegated, parent_wallet = entry
        try:
            returned = await release_delegated_wallet(delegated, parent_wallet)
            logger.info(
                "Released delegated budget for '%s': returned %s to parent (#2113).",
                child_name, returned,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to release delegated budget for '%s': %s", child_name, e
            )

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
        # Subset-of-parent validation (F277): a mandate must only ever RESTRICT
        # the child relative to the parent — it may never grant features the
        # parent lacks or add constitution-weakening constraints. Enforce this
        # before any inception work, so an over-broad mandate is refused rather
        # than silently producing a child with more authority than its parent.
        self._validate_mandate_subset(parent_agent, mandate)

        # Budget precondition (#2113): a positive budget requires a funded parent
        # wallet to hold from, or the ceiling would be advertised-but-unenforced.
        # Validated before any inception work so it is refused rather than
        # producing an uncapped child.
        self._validate_budget_precondition(parent_agent, mandate)

        # Spawn caps (#1729): bound runaway spawning. The check + reservation run
        # under the manager lock so concurrent spawn_agent calls can't all read
        # the same count and blow past the cap (codex r2). ``_pending_spawns``
        # counts in-flight spawns whose mandate isn't registered yet.
        async with self._lock:
            in_use = len(self._child_mandates) + self._pending_spawns
            if in_use >= self._max_spawned_agents:
                raise ValueError(
                    f"Spawn refused: at the spawned-agent cap "
                    f"({self._max_spawned_agents}). Set KESTREL_MAX_SPAWNED_AGENTS to raise."
                )
            # Depth cap — if the PARENT was itself spawned and its mandate marks
            # it a leaf (max_child_depth <= 0), it may not spawn further.
            parent_name = self._agent_names.get(parent_agent.agent_id)
            parent_mandate = self._child_mandates.get(parent_name) if parent_name else None
            if parent_mandate is not None and getattr(parent_mandate, "max_child_depth", 0) <= 0:
                raise ValueError(
                    f"Spawn refused: parent '{parent_name}' is at its max child depth "
                    f"(mandate max_child_depth={getattr(parent_mandate, 'max_child_depth', 0)})."
                )
            self._pending_spawns += 1

        try:
            # DECREMENT remaining depth on delegation (codex r2): a non-leaf
            # spawned parent's child must have strictly less depth, regardless of
            # what the caller put in the mandate — otherwise depth never shrinks.
            if parent_mandate is not None:
                allowed = getattr(parent_mandate, "max_child_depth", 0) - 1
                if getattr(mandate, "max_child_depth", 0) > allowed:
                    mandate.max_child_depth = max(allowed, 0)
            return await self._do_spawn(name, parent_agent, mandate)
        finally:
            async with self._lock:
                self._pending_spawns -= 1

    async def _do_spawn(
        self,
        name: str,
        parent_agent: KestrelAgent,
        mandate: SpawnMandate,
    ) -> KestrelAgent:
        """Sign the mandate, create the child, and register it (#1729)."""
        # Sign the mandate with the parent's keys if available.
        # Hybrid parents (post-rotation ceremony) get an additional
        # ``parent_identity`` arg so the mandate is signed with both
        # Ed25519 and ML-DSA-65; legacy parents fall through to the
        # bare-hex ECDSA path. The parent's runtime identity is set
        # on the agent at startup by KestrelAgent.__init__ (#999).
        # Resolve the child's feature ceiling BEFORE signing so the signed
        # mandate — and the spawned_by edge inception persists from it — records
        # the ACTUAL ceiling, explicit or inherited (#1946). An empty list is a
        # real (empty) allowlist; only ``None`` means "load all".
        explicit_features = getattr(mandate, "features_allowed", None)
        if explicit_features:
            child_features = list(explicit_features)
        else:
            # No explicit allowlist ⇒ inherit the PARENT's feature ceiling, NOT
            # "load all discovered features" (F277 / codex P1). Otherwise a
            # restricted parent could spawn a broader-than-itself child simply by
            # omitting features_allowed. A parent with no resolvable feature set
            # (degenerate/test doubles) falls back to None (load all).
            child_features = sorted(self._parent_feature_names(parent_agent)) or None
            # Persist the INHERITED ceiling onto the mandate so it is durable on
            # the edge and enforced on every boot path (#2226) — not just via
            # this process's config threading. Without this, an inherited-ceiling
            # child persists an empty features_allowed and, on a direct restart
            # outside AgentManager, would escape its ceiling and load everything.
            if child_features:
                mandate.features_allowed = list(child_features)

        parent_private_key = getattr(parent_agent, '_private_key', None)
        parent_identity = getattr(parent_agent, 'identity', None)
        if parent_private_key is not None:
            sign_mandate(
                mandate, parent_private_key,
                parent_identity=parent_identity,
            )
        child = await self.create_agent(
            name,
            parent_did=parent_agent.agent_id,
            features=child_features,
            mandate=mandate,
        )

        # Fill in child DID on the mandate
        mandate.child_did = child.agent_id

        # Per-child budget (#2113): hold from the parent and route the child's
        # spend through a ceiling'd DelegatedWallet. On hold failure this removes
        # the just-created child and re-raises (no uncapped orphan).
        await self._apply_delegated_budget(name, parent_agent, child, mandate)

        # Runtime enforcement (spawn_mandate attach + restricted_tools hook) is
        # applied uniformly in load_agent from the persisted delegation edge
        # (#2137), which already ran for this child inside create_agent — so it
        # covers reload/restart, not just this in-process spawn.

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

        # NB: the child's own budget hold is released inside remove_agent below
        # (stop-then-release), after the cascade above has already stopped and
        # released every descendant — so refunds flow up leaf-first (#2113).

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
        # Stop + release budgeted children leaf-first (#2113): reverse insertion
        # order is leaf-first (a descendant is always spawned after its ancestor),
        # so each is quiesced and its unspent hold refunded UP into its (budgeted)
        # parent before that parent is released to the root. remove_agent does the
        # stop-then-release. (A follow-up covers durable reconciliation across an
        # *ungraceful* crash.)
        for child_name in reversed(list(self._child_budgets.keys())):
            await self.remove_agent(child_name)

        # Clear parent-child tracking
        self._parent_children.clear()
        self._child_mandates.clear()

        names = list(self._agents.keys())
        for name in names:
            await self.remove_agent(name)
        logger.info("All agents shut down")
