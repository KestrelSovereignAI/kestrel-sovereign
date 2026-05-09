"""
Spawn Feature — Runtime agent spawning as LLM-callable tools.

Allows the orchestrator LLM to autonomously create child agents,
delegate tasks, retrieve results, and terminate children. Built on
top of the AgentManager's spawn primitives and SpawnMandate for
cryptographic delegation chains.

Architecture:
    Orchestrator LLM → SpawnFeature.spawn_agent(name, purpose, ...)
        → AgentManager.spawn_agent(name, parent_agent, mandate)
            → Inception → Child KestrelAgent running in-process
        ← Child DID returned
    Orchestrator LLM → SpawnFeature.delegate_task(child_name, task)
        → Child agent processes task via its own LLM context
        ← Result stored for later retrieval
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

logger = logging.getLogger(__name__)


class SpawnFeature(Feature):
    """Runtime agent spawning — create, delegate to, and manage child agents."""

    @property
    def tool_description(self) -> str:
        return (
            "Spawn and manage child agents — create new agents with specific purposes, "
            "delegate tasks, retrieve results, and terminate children"
        )

    async def initialize(self):
        self._agent_manager = None
        self._lifecycle = None
        self._child_results: dict[str, Any] = {}  # child_name -> latest result
        self._child_tasks: dict[str, asyncio.Task] = {}  # child_name -> running task

    def get_router(self):
        """Return the Spawn panel router for dynamic mounting.

        The router is defined in endpoints/spawn.py and mounted by the
        server only when SpawnFeature is discovered and enabled.
        """
        from kestrel_sovereign.endpoints.spawn import router
        return router

    async def post_all_features_loaded(self, agent):
        """Pre-explore spawn tools so they are immediately available to the orchestrator."""
        if hasattr(agent, '_register_explored_feature_tools'):
            agent._register_explored_feature_tools(self)
            logger.info("SpawnFeature tools pre-explored for direct calling")

    def _get_agent_manager(self):
        """Lazily resolve or create an AgentManager.

        In multi_agent mode the manager already exists on the agent.  In
        single-agent mode we create a lightweight one on-the-fly so
        spawn_agent works regardless of deployment mode.
        """
        if self._agent_manager is not None:
            return self._agent_manager

        # Try to get from the agent's registered manager
        manager = getattr(self.agent, '_agent_manager', None)
        if manager is None:
            manager = getattr(self.agent, 'agent_manager', None)

        # Single-agent mode: create a lightweight AgentManager
        if manager is None:
            from kestrel_sovereign.multi_agent.agent_manager import AgentManager
            storage_dir = getattr(self.agent, 'storage_path', None)
            base_dir = Path(storage_dir).parent.parent if storage_dir else None
            manager = AgentManager(base_data_dir=base_dir)
            # Register the current agent so it appears as the parent
            agent_name = getattr(self.agent, 'agent_name', None) or 'default'
            manager._agents[agent_name] = self.agent
            agent_did = getattr(self.agent, 'did', None) or ''
            if agent_did:
                manager._agent_names[agent_did] = agent_name
            # Attach back to agent so endpoints and other code can find it
            self.agent._agent_manager = manager
            logger.info("Created lightweight AgentManager for single-agent spawn")

        # Ensure lifecycle is wired up
        if not getattr(manager, '_lifecycle', None):
            from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle
            lifecycle = SpawnedAgentLifecycle(manager)
            manager._lifecycle = lifecycle
            self._lifecycle = lifecycle
            logger.info("SpawnedAgentLifecycle attached to AgentManager")

        self._agent_manager = manager
        return manager

    def _set_agent_manager(self, manager):
        """Explicitly set the AgentManager (used in testing)."""
        self._agent_manager = manager

    def _get_lifecycle(self, manager):
        """Return the lifecycle manager when it is fully wired."""
        from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle

        lifecycle = getattr(manager, "_lifecycle", None)
        if isinstance(lifecycle, SpawnedAgentLifecycle):
            return lifecycle
        return None

    @tool(
        name="spawn_agent",
        description=(
            "Create a new child agent with a specific purpose and constraints. "
            "The child runs in-process and is governed by a signed SpawnMandate. "
            "Returns the child's name and DID on success."
        ),
        category=ToolCategory.AGENT_MANAGEMENT,
    )
    async def spawn_agent(
        self,
        name: str,
        purpose: str,
        budget: float = 0.0,
        ttl: int = 3600,
        constraints: str = "",
        features: str = "",
    ) -> ToolResult:
        """
        Create a child agent with a signed mandate.

        Args:
            name: Unique name for the child agent
            purpose: What the child agent is for (stored in mandate)
            budget: Budget allocation for the child (default 0)
            ttl: Time-to-live in seconds (default 3600)
            constraints: Comma-separated additional constraints
            features: Comma-separated list of allowed features

        Returns:
            ToolResult.ok with child name + DID + purpose + TTL.
            ERROR when no AgentManager is wired up (agent isn't
            running in a multi-agent host) or the spawn raised.
        """
        manager = self._get_agent_manager()
        if manager is None:
            return ToolResult.failed(
                error="No AgentManager available — agent is not running in a multi_agent"
            )

        # Parse comma-separated strings into lists
        constraint_dict = {}
        if constraints:
            for item in constraints.split(","):
                item = item.strip()
                if "=" in item:
                    k, v = item.split("=", 1)
                    constraint_dict[k.strip()] = v.strip()
                elif item:
                    constraint_dict[item] = "true"

        features_list = [f.strip() for f in features.split(",") if f.strip()] if features else []

        # Build the mandate
        from kestrel_sovereign.spawn.mandate import SpawnMandate

        mandate = SpawnMandate(
            parent_did=self.agent.agent_id,
            purpose=purpose,
            budget_allocation=budget,
            ttl_seconds=ttl,
            additional_constraints=constraint_dict,
            features_allowed=features_list,
        )

        try:
            child = await manager.spawn_agent(
                name=name,
                parent_agent=self.agent,
                mandate=mandate,
            )
            # Register with lifecycle for TTL tracking and history
            from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle
            lifecycle = getattr(manager, '_lifecycle', None)
            if isinstance(lifecycle, SpawnedAgentLifecycle):
                from kestrel_sovereign.spawn.lifecycle import SpawnMode
                await lifecycle.register(
                    child_name=name,
                    child_did=child.agent_id,
                    parent_did=self.agent.agent_id,
                    ttl_seconds=ttl,
                    purpose=purpose,
                    mode=SpawnMode.EPHEMERAL if ttl > 0 else SpawnMode.PERSISTENT,
                )
            return ToolResult.ok(
                f"Spawned child '{name}' (did={child.agent_id}, ttl={ttl}s).",
                data={
                    "spawned": True,
                    "child_name": name,
                    "child_did": child.agent_id,
                    "purpose": purpose,
                    "ttl": ttl,
                },
            )
        except Exception as e:
            logger.error(f"Failed to spawn child agent '{name}': {e}")
            return ToolResult.failed(error=str(e))

    @tool(
        name="list_children",
        description="List all active child agents spawned by this agent, with their status.",
        category=ToolCategory.AGENT_MANAGEMENT,
    )
    async def list_children(self) -> ToolResult:
        """
        List active children with status.

        Returns:
            ToolResult.ok with the list of children and their status.
            ERROR when there's no AgentManager (agent isn't in a
            multi-agent host) — was the legacy
            ``{"children": [], "error": "..."}`` shape that mixed
            empty-list-of-children with the no-host-error case.
        """
        manager = self._get_agent_manager()
        if manager is None:
            return ToolResult.failed(error="No AgentManager available")

        parent_did = self.agent.agent_id
        child_names = manager.get_children(parent_did)

        children = []
        for child_name in child_names:
            child_agent = manager.get_agent(child_name)
            status = "running" if child_agent is not None else "stopped"

            has_result = child_name in self._child_results
            has_pending_task = (
                child_name in self._child_tasks
                and not self._child_tasks[child_name].done()
            )

            children.append({
                "name": child_name,
                "status": status,
                "has_result": has_result,
                "has_pending_task": has_pending_task,
            })

        if not children:
            return ToolResult.ok(
                "No active children.",
                data={"children": [], "count": 0},
            )
        return ToolResult.ok(
            f"{len(children)} active child(ren): "
            + ", ".join(f"{c['name']} ({c['status']})" for c in children)
            + ".",
            data={"children": children, "count": len(children)},
        )

    @tool(
        name="delegate_task",
        description=(
            "Send a task to an existing child agent for processing. "
            "The child uses its own LLM context to execute the task. "
            "Results can be retrieved later with get_child_result."
        ),
        category=ToolCategory.AGENT_MANAGEMENT,
    )
    async def delegate_task(self, child_name: str, task: str) -> ToolResult:
        """
        Send work to an existing child agent.

        Args:
            child_name: Name of the child agent to delegate to
            task: The task description for the child to execute

        Returns:
            ToolResult.ok with the queued task info and a note that
            ``get_child_result`` is the way to retrieve the output.
            ERROR for no-AgentManager, child-not-running, and
            child-not-belonging-to-this-parent paths.
        """
        manager = self._get_agent_manager()
        if manager is None:
            return ToolResult.failed(error="No AgentManager available")

        child_agent = manager.get_agent(child_name)
        if child_agent is None:
            return ToolResult.failed(
                error=f"Child agent '{child_name}' not found or not running"
            )

        # Verify this is actually our child
        parent_did = self.agent.agent_id
        if child_name not in manager.get_children(parent_did):
            return ToolResult.failed(
                error=f"Agent '{child_name}' is not a child of this agent"
            )

        # Run the task asynchronously via the child agent's chat method
        async def _run_child_task():
            lifecycle = self._get_lifecycle(manager)
            try:
                result = await child_agent.process_input(task)
                self._child_results[child_name] = {
                    "success": True,
                    "result": result,
                    "completed_at": time.time(),
                }
                if lifecycle is not None:
                    await lifecycle.report_result(
                        child_name=child_name,
                        output_artifacts={"result": result},
                    )
            except Exception as e:
                logger.error(f"Child '{child_name}' task failed: {e}")
                self._child_results[child_name] = {
                    "success": False,
                    "error": str(e),
                    "completed_at": time.time(),
                }
                if lifecycle is not None:
                    from kestrel_sovereign.spawn.lifecycle import SpawnStatus
                    await lifecycle.report_result(
                        child_name=child_name,
                        output_artifacts={"error": str(e)},
                        status=SpawnStatus.FAILED,
                    )

        # Cancel any existing task for this child
        if child_name in self._child_tasks and not self._child_tasks[child_name].done():
            self._child_tasks[child_name].cancel()

        self._child_tasks[child_name] = asyncio.create_task(_run_child_task())

        return ToolResult.ok(
            f"Delegated task to '{child_name}'. Use get_child_result to retrieve the result.",
            data={
                "delegated": True,
                "child_name": child_name,
                "task": task,
                "note": "Task is running. Use get_child_result to retrieve the result.",
            },
        )

    @tool(
        name="get_child_result",
        description="Retrieve the result from a child agent's completed task.",
        category=ToolCategory.AGENT_MANAGEMENT,
    )
    async def get_child_result(self, child_name: str) -> ToolResult:
        """
        Retrieve results from a completed child task.

        Args:
            child_name: Name of the child agent to get results from

        Returns:
            ToolResult.ok with ready=True + the child's task output
            on success. PARTIAL when the task is still running OR no
            result is available — these are not errors (the LLM may
            legitimately need to poll), but the LLM must NOT report
            them as completed work; the caveat says exactly which
            "not ready" state we're in.
            ERROR when the underlying child task itself raised — the
            stored failure dict carries ``error`` and the envelope
            ERROR copies it through so the LLM speaks the failure
            instead of treating "ready=True" as success.
        """
        # Check if there's a pending task still running
        if child_name in self._child_tasks and not self._child_tasks[child_name].done():
            return ToolResult.partial(
                f"Task for '{child_name}' is still running.",
                "no result yet — try get_child_result again after the task completes.",
                data={
                    "ready": False,
                    "child_name": child_name,
                    "note": "Task is still running. Try again later.",
                },
            )

        if child_name not in self._child_results:
            return ToolResult.partial(
                f"No result available for '{child_name}'.",
                (
                    "either no task was delegated to this child, or the "
                    "previous result was already consumed by a prior "
                    "get_child_result call (this surface pops on read)."
                ),
                data={
                    "ready": False,
                    "child_name": child_name,
                    "note": "No result available. Either no task was delegated or it hasn't completed.",
                },
            )

        result = self._child_results.pop(child_name)
        # The stored result has ``success`` and either ``result`` (on
        # success) or ``error`` (on failure). Surface child-side
        # failures as ERROR so the LLM doesn't claim the delegated
        # work succeeded.
        if not result.get("success"):
            err = result.get("error") or f"Child task for '{child_name}' failed"
            return ToolResult.failed(
                error=err,
                data={"ready": True, "child_name": child_name, **result},
            )
        return ToolResult.ok(
            f"Retrieved result for '{child_name}'.",
            data={"ready": True, "child_name": child_name, **result},
        )

    @tool(
        name="terminate_child",
        description="Terminate a child agent, stopping it and releasing its resources.",
        category=ToolCategory.AGENT_MANAGEMENT,
    )
    async def terminate_child(self, child_name: str) -> ToolResult:
        """
        Terminate a child agent early.

        Args:
            child_name: Name of the child agent to terminate

        Returns:
            ToolResult.ok when the child was actually terminated.
            ERROR when there's no AgentManager, the named agent is
            not a child of this parent, or the underlying terminate
            call returned False.
        """
        manager = self._get_agent_manager()
        if manager is None:
            return ToolResult.failed(error="No AgentManager available")

        # Verify this is our child
        parent_did = self.agent.agent_id
        if child_name not in manager.get_children(parent_did):
            return ToolResult.failed(
                error=f"Agent '{child_name}' is not a child of this agent"
            )

        # Cancel any running task
        if child_name in self._child_tasks and not self._child_tasks[child_name].done():
            self._child_tasks[child_name].cancel()
            self._child_tasks.pop(child_name, None)

        # Clean up stored results
        self._child_results.pop(child_name, None)

        # Remove from manager (handles shutdown + cascading child termination)
        lifecycle = self._get_lifecycle(manager)
        if lifecycle is not None:
            result = await lifecycle.terminate(
                child_name=child_name,
                reason="explicit termination",
            )
            removed = result is not None
        else:
            removed = await manager.terminate_child(parent_did, child_name)
        if removed:
            return ToolResult.ok(
                f"Terminated child '{child_name}'.",
                data={"terminated": True, "child_name": child_name},
            )

        return ToolResult.failed(error=f"Failed to terminate '{child_name}'")

    async def shutdown(self):
        """Clean up running tasks on feature shutdown."""
        for task in self._child_tasks.values():
            if not task.done():
                task.cancel()
        self._child_tasks.clear()
        self._child_results.clear()

    @property
    def default_permissions(self) -> Dict[str, str]:
        """Default permission levels for spawn tools.

        spawn_agent requires explicit approval (ASK) since it creates new agents.
        Other tools default to ALLOW since they operate on already-approved children.
        """
        return {
            "spawn_agent": "ask",
            "list_children": "allow",
            "delegate_task": "allow",
            "get_child_result": "allow",
            "terminate_child": "allow",
        }
