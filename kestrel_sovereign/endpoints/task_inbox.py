"""Operator inbox presentation, separate from native A2A protocol routes."""

from fastapi import APIRouter, HTTPException, Query, Request
from google.protobuf.json_format import MessageToDict

from kestrel_sovereign.a2a.lifecycle import task_state_from_label
from kestrel_sovereign.a2a.projections import task_presentation
from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.endpoints.agent_helpers import get_agent
from kestrel_sovereign.features.storage_access import (
    AgentIdentityUnavailable,
    resolve_scoped_agent_did,
)

router = APIRouter(prefix="/api/agent/task-inbox", tags=["task-inbox"])


def _manager(request):
    agent = get_agent(request)
    manager = getattr(agent, "task_manager", None)
    if not isinstance(manager, TaskManager):
        raise HTTPException(503, "Native task inbox is not available")
    try:
        recipient = resolve_scoped_agent_did(agent)
    except AgentIdentityUnavailable as error:
        raise HTTPException(
            503, "Native task inbox requires a durable recipient identity"
        ) from error
    if manager.task_store.scope.recipient_id != recipient:
        raise HTTPException(503, "Native task inbox does not belong to this agent")
    return manager, recipient


@router.get("")
async def list_inbox(
    request: Request, status: str | None = None, limit: int = Query(50, ge=1, le=100)
):
    manager, recipient = _manager(request)
    try:
        state = task_state_from_label(status) if status is not None else None
    except ValueError as error:
        raise HTTPException(400, "Unknown native task status") from error
    candidates = await manager.list_task_snapshots(
        recipient_agent_id=recipient, status=state, limit=limit
    )
    entries = [
        task_presentation(candidate, include_artifacts=False)
        for candidate in candidates
    ]
    return {"tasks": entries, "total": len(entries)}


@router.get("/{task_id}")
async def read_inbox(request: Request, task_id: str):
    manager, recipient = _manager(request)
    candidate = await manager.get_task_snapshot(task_id, recipient_agent_id=recipient)
    if candidate is None:
        raise HTTPException(404, "Native task not found in this recipient's inbox")
    return {
        "summary": task_presentation(candidate),
        "task": MessageToDict(candidate.task),
    }
