"""Spawn panel API endpoints — real-time spawn state for the Console UI.

Inlined into core after the external ``kestrel-feature-spawn``
package was archived 2026-04-05. Originally re-exported from
``kestrel_feature_spawn.spawn.endpoints`` via a try/except import
shim that produced the ``Failed to mount router from feature
SpawnFeature: No module named 'kestrel_feature_spawn'`` boot
warning. The shim is gone; this module is now the canonical
location for the spawn panel routes.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request

from kestrel_sovereign.endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/spawn", tags=["spawn"])


def _get_lifecycle(agent, request: Request | None = None) -> Any:
    """Get the SpawnedAgentLifecycle from the agent manager, if available.

    Falls back to ``request.app.state.agent_manager`` for the same
    multi-agent reason called out in ``_get_agent_manager``.
    """
    manager = getattr(agent, '_agent_manager', None) or getattr(agent, 'agent_manager', None)
    if manager is None and request is not None:
        manager = getattr(request.app.state, 'agent_manager', None)
    if manager is None:
        return None
    return getattr(manager, '_lifecycle', None)


def _get_agent_manager(agent, request: Request | None = None) -> Any:
    """Get AgentManager — first from the agent, then from app.state.

    In multi-agent mode the shared AgentManager lives on
    ``request.app.state.agent_manager`` and is NOT attached to each
    loaded agent (``AgentManager.load_agent()`` doesn't backref).
    Without the request fallback the panel reports an empty spawn
    state for routed-agent calls like ``/api/agents/{name}/api/spawn/children``
    even when the app-level manager has children. Codex round 3 of
    #1149 caught this; the bug existed in the archived package too.
    """
    manager = getattr(agent, '_agent_manager', None) or getattr(agent, 'agent_manager', None)
    if manager is None and request is not None:
        manager = getattr(request.app.state, 'agent_manager', None)
    return manager


@router.get("/children")
async def get_spawn_children(request: Request):
    """List all spawned children with status, budget, TTL, and delegation chain."""
    agent = get_agent(request)
    manager = _get_agent_manager(agent, request=request)

    if manager is None:
        return {"children": [], "count": 0, "delegation_chain": {}, "history": []}

    parent_did = agent.agent_id
    child_names = manager.get_children(parent_did)

    children = []
    now = datetime.now(timezone.utc)

    for child_name in child_names:
        child_agent = manager.get_agent(child_name)
        mandate = manager.get_mandate(child_name)

        child_info = {
            "name": child_name,
            "status": "running" if child_agent is not None else "stopped",
            "did": child_agent.agent_id if child_agent else "",
            "purpose": mandate.purpose if mandate else "",
            "ttl_seconds": mandate.ttl_seconds if mandate else 0,
            "budget_allocated": float(mandate.budget_allocation) if mandate else 0.0,
            "budget_spent": 0.0,
            "budget_remaining": float(mandate.budget_allocation) if mandate else 0.0,
            "started_at": mandate.created_at if mandate else "",
            "ttl_remaining": 0,
        }

        # Calculate TTL remaining
        if mandate and mandate.created_at:
            try:
                started = datetime.fromisoformat(mandate.created_at)
                elapsed = (now - started).total_seconds()
                child_info["ttl_remaining"] = max(0, mandate.ttl_seconds - elapsed)
            except (ValueError, TypeError):
                pass

        # Try to get budget info from lifecycle tracker
        lifecycle = _get_lifecycle(agent, request=request)
        if lifecycle is not None:
            tracked = lifecycle._tracked.get(child_name)
            if tracked and tracked.result:
                child_info["budget_spent"] = float(tracked.result.budget_consumed)
                child_info["budget_remaining"] = child_info["budget_allocated"] - child_info["budget_spent"]

        # Get delegated wallet info if available
        if child_agent is not None:
            wallet = getattr(child_agent, '_delegated_wallet', None)
            if wallet is not None:
                allocation = getattr(wallet, '_allocation', None)
                if allocation is not None:
                    child_info["budget_spent"] = float(allocation.spent)
                    child_info["budget_remaining"] = float(allocation.remaining)

        children.append(child_info)

    # Build delegation chain tree
    delegation_chain = _build_delegation_chain(manager, parent_did, agent.agent_id)

    # Build spawn history from lifecycle results
    history = _build_spawn_history(agent, manager, request=request)

    return {
        "children": children,
        "count": len(children),
        "delegation_chain": delegation_chain,
        "history": history,
    }


def _build_delegation_chain(manager, parent_did: str, parent_name: str) -> dict:
    """Build a tree structure showing Parent -> Child -> Grandchild relationships."""
    child_names = manager.get_children(parent_did)
    children_nodes = []

    for child_name in child_names:
        child_agent = manager.get_agent(child_name)
        child_did = child_agent.agent_id if child_agent else ""
        mandate = manager.get_mandate(child_name)

        child_node = {
            "name": child_name,
            "did": child_did,
            "purpose": mandate.purpose if mandate else "",
            "status": "running" if child_agent is not None else "stopped",
            "children": [],
        }

        # Recurse for grandchildren
        if child_did:
            grandchildren = manager.get_children(child_did)
            if grandchildren:
                child_node["children"] = _build_delegation_chain(
                    manager, child_did, child_name
                ).get("children", [])

        children_nodes.append(child_node)

    return {
        "name": parent_name,
        "did": parent_did,
        "status": "running",
        "children": children_nodes,
    }


def _build_spawn_history(agent, manager, request: Request | None = None) -> list:
    """Build a log of spawn/terminate events for THIS agent only.

    In multi-agent mode the lifecycle is attached to the shared
    ``app.state.agent_manager``, so its ``_tracked`` and
    ``_results`` dicts contain children spawned by every loaded
    parent. We filter both by ``parent_did == agent.agent_id`` so
    one agent's panel never leaks another agent's history (#1149
    round 4 caught this).
    """
    history = []
    parent_did = agent.agent_id

    lifecycle = _get_lifecycle(agent, request=request)
    if lifecycle is not None:
        # Completed/terminated results — filter by parent_did.
        # SpawnResult.parent_did was added in #1149 round 4. Old
        # serialized records may lack it (defaulted to ""); when
        # that's the case we fall through to the active-tracked
        # cross-reference below.
        for name, result in lifecycle._results.items():
            result_parent = getattr(result, "parent_did", "") or ""
            if result_parent and result_parent != parent_did:
                continue
            history.append({
                "event": "terminated",
                "child_name": result.child_name,
                "child_did": result.child_did,
                "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                "budget_consumed": float(result.budget_consumed),
                "started_at": result.started_at,
                "ended_at": result.ended_at,
            })

        # Currently tracked (active spawns) — _TrackedChild always
        # has parent_did, so filtering is unconditional.
        for name, tracked in lifecycle._tracked.items():
            if tracked.parent_did != parent_did:
                continue
            history.append({
                "event": "spawned",
                "child_name": tracked.child_name,
                "child_did": tracked.child_did,
                "status": "running",
                "started_at": tracked.started_at,
            })

    # Sort by timestamp, most recent first
    history.sort(key=lambda h: h.get("started_at", ""), reverse=True)
    return history


__all__ = ["router"]
