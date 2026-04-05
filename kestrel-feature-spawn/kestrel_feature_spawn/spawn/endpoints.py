"""Spawn panel API endpoints — real-time spawn state for the Console UI."""

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request

from endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/spawn", tags=["spawn"])


def _get_lifecycle(agent) -> Any:
    """Get the SpawnedAgentLifecycle from the agent manager, if available."""
    manager = getattr(agent, '_agent_manager', None) or getattr(agent, 'agent_manager', None)
    if manager is None:
        return None
    return getattr(manager, '_lifecycle', None)


def _get_agent_manager(agent) -> Any:
    """Get AgentManager from agent."""
    return getattr(agent, '_agent_manager', None) or getattr(agent, 'agent_manager', None)


@router.get("/children")
async def get_spawn_children(request: Request):
    """List all spawned children with status, budget, TTL, and delegation chain."""
    agent = get_agent(request)
    manager = _get_agent_manager(agent)

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
        lifecycle = _get_lifecycle(agent)
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
    history = _build_spawn_history(agent, manager)

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


def _build_spawn_history(agent, manager) -> list:
    """Build a log of spawn/terminate events."""
    history = []

    lifecycle = _get_lifecycle(agent)
    if lifecycle is not None:
        # Completed/terminated results
        for name, result in lifecycle._results.items():
            history.append({
                "event": "terminated",
                "child_name": result.child_name,
                "child_did": result.child_did,
                "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                "budget_consumed": float(result.budget_consumed),
                "started_at": result.started_at,
                "ended_at": result.ended_at,
            })

        # Currently tracked (active spawns)
        for name, tracked in lifecycle._tracked.items():
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
