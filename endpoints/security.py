"""
Kestrel Security API Endpoints.

Provides REST API for managing security permissions and approval queue.
"""

from fastapi import APIRouter, Query, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from kestrel_sovereign.rate_limit import limiter

router = APIRouter(prefix="/api/security", tags=["security"])


# === Request/Response Models ===

class SetPermissionRequest(BaseModel):
    """Request to set a tool permission."""
    feature: str
    tool: Optional[str] = None
    level: str  # "allow", "deny", "ask", "session"


class SetFeaturePermissionRequest(BaseModel):
    """Request to set all tools in a feature."""
    feature: str
    level: str


class ApprovalDecisionRequest(BaseModel):
    """Request to submit approval decision."""
    approval_id: str
    approved: bool
    scope: str = "once"  # "once", "session", "always"


class ToolPermissionResponse(BaseModel):
    """Single tool permission."""
    name: str
    level: str


class FeaturePermissionResponse(BaseModel):
    """Feature with tools and rollup state."""
    name: str
    rollup_state: str
    tools: List[ToolPermissionResponse]


class PermissionTreeResponse(BaseModel):
    """Full permission tree."""
    tree: List[FeaturePermissionResponse]


class PendingApprovalResponse(BaseModel):
    """Pending approval request."""
    id: str
    feature: str
    tool: str
    args: Dict[str, Any]
    timestamp: str


class PendingListResponse(BaseModel):
    """List of pending approvals."""
    pending: List[PendingApprovalResponse]
    count: int


class AuditLogEntry(BaseModel):
    """Single audit log entry."""
    feature: str
    tool: str
    action: str
    decision: str
    user_choice: Optional[str]
    args_summary: Optional[str]
    timestamp: str


class AuditLogResponse(BaseModel):
    """Audit log response."""
    logs: List[AuditLogEntry]


# === Helper Functions ===

def get_security_feature(request: Request):
    """
    Get SecurityFeature from kestrel_sovereign.agent.

    Raises:
        HTTPException: If security feature not available
    """
    agent = getattr(request.app.state, "agent", None)
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    security = agent.features.get("SecurityFeature")
    if not security:
        raise HTTPException(
            status_code=503,
            detail="SecurityFeature not available"
        )

    return security


# === Endpoints ===

@router.get("/permissions/tree", response_model=PermissionTreeResponse)
async def get_permission_tree(request: Request):
    """
    Get hierarchical permission tree for UI.

    Returns the full tree of features and tools with their
    permission levels and rollup states.
    """
    security = get_security_feature(request)
    tree = await security.permission_store.get_permission_tree()

    return PermissionTreeResponse(
        tree=[
            FeaturePermissionResponse(
                name=f.feature_name,
                rollup_state=f.rollup_state,
                tools=[
                    ToolPermissionResponse(
                        name=t.tool_name,
                        level=t.level.value
                    )
                    for t in f.tools
                ]
            )
            for f in tree
        ]
    )


@router.post("/permissions")
@limiter.limit("30/minute")
async def set_tool_permission(request: Request, data: SetPermissionRequest):
    """
    Set permission for a specific tool.

    If tool is not specified, sets permission for all tools in the feature.
    """
    security = get_security_feature(request)

    try:
        from kestrel_sovereign.features.security.permissions import PermissionLevel
        level = PermissionLevel(data.level)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid level '{data.level}'. Use: allow, deny, ask, session"
        )

    if data.tool:
        await security.permission_store.set_permission(
            data.feature, data.tool, level
        )
        return {"success": True, "message": f"Set {data.feature}.{data.tool} to {data.level}"}
    else:
        await security.permission_store.set_feature_permission(
            data.feature, level
        )
        return {"success": True, "message": f"Set all tools in {data.feature} to {data.level}"}


@router.post("/permissions/feature")
@limiter.limit("30/minute")
async def set_feature_permission(request: Request, data: SetFeaturePermissionRequest):
    """
    Set permission for all tools in a feature (bulk update).

    This is the same as calling POST /permissions without a tool name.
    """
    security = get_security_feature(request)

    try:
        from kestrel_sovereign.features.security.permissions import PermissionLevel
        level = PermissionLevel(data.level)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid level '{data.level}'. Use: allow, deny, ask, session"
        )

    await security.permission_store.set_feature_permission(data.feature, level)
    return {"success": True, "message": f"Set all tools in {data.feature} to {data.level}"}


@router.get("/pending", response_model=PendingListResponse)
async def get_pending_approvals(request: Request):
    """
    Get pending approval requests.

    Returns list of requests waiting for user decision.
    """
    security = get_security_feature(request)
    pending = security.approval_queue.pending_requests

    return PendingListResponse(
        pending=[
            PendingApprovalResponse(
                id=r.id,
                feature=r.feature_name,
                tool=r.tool_name,
                args=r.tool_args,
                timestamp=r.created_at.isoformat()
            )
            for r in pending
        ],
        count=len(pending)
    )


@router.post("/approve")
@limiter.limit("30/minute")
async def submit_approval(request: Request, data: ApprovalDecisionRequest):
    """
    Submit approval decision for a pending request.

    Args:
        approval_id: ID of the pending request
        approved: Whether to approve or deny
        scope: "once", "session", or "always"
    """
    security = get_security_feature(request)

    if data.scope not in ("once", "session", "always"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scope '{data.scope}'. Use: once, session, always"
        )

    success = security.approval_queue.submit_decision(
        data.approval_id, data.approved, data.scope
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Request '{data.approval_id}' not found or expired"
        )

    return {
        "success": True,
        "approved": data.approved,
        "scope": data.scope
    }


@router.get("/audit", response_model=AuditLogResponse)
async def get_audit_log(request: Request, limit: int = Query(50, ge=1, le=500)):
    """
    Get security audit log.

    Returns recent permission decisions and user choices.
    """
    security = get_security_feature(request)

    logs = await security.permission_store.get_audit_log(limit)

    return AuditLogResponse(
        logs=[
            AuditLogEntry(
                feature=log["feature"],
                tool=log["tool"],
                action=log["action"],
                decision=log["decision"],
                user_choice=log["user_choice"],
                args_summary=log["args_summary"],
                timestamp=log["timestamp"]
            )
            for log in logs
        ]
    )


@router.post("/cancel/{request_id}")
@limiter.limit("30/minute")
async def cancel_request(request: Request, request_id: str):
    """
    Cancel a pending approval request.

    The tool execution waiting for approval will be denied.
    """
    security = get_security_feature(request)

    if security.approval_queue.cancel_request(request_id):
        return {"success": True, "message": f"Cancelled request {request_id[:8]}"}

    raise HTTPException(
        status_code=404,
        detail=f"Request '{request_id}' not found"
    )


@router.post("/cancel-all")
@limiter.limit("30/minute")
async def cancel_all_requests(request: Request):
    """
    Cancel all pending approval requests.

    All pending tool executions will be denied.
    """
    security = get_security_feature(request)
    count = security.approval_queue.cancel_all()
    return {"success": True, "cancelled": count}


@router.post("/reset-session")
@limiter.limit("30/minute")
async def reset_session_overrides(request: Request):
    """
    Clear all session-scoped permission overrides.

    Tools that were approved for "this session" will revert
    to requiring approval again.
    """
    security = get_security_feature(request)
    security.permission_store.clear_session_overrides()
    return {"success": True, "message": "Session overrides cleared"}
