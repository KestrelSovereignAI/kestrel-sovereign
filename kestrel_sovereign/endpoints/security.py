"""
Kestrel Security API Endpoints.

Provides REST API for managing security permissions and approval queue.
"""

import logging

from fastapi import APIRouter, Query, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from kestrel_sovereign.rate_limit import limiter
from kestrel_sovereign.endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/security", tags=["security"])


# === Request/Response Models ===

class SetPermissionRequest(BaseModel):
    """Request to set a tool permission."""
    feature: str
    tool: Optional[str] = None
    level: str  # "allow", "auto", "always_ask", "deny", "ask", "session"


class SetFeaturePermissionRequest(BaseModel):
    """Request to set all tools in a feature."""
    feature: str
    level: str


class ApprovalDecisionRequest(BaseModel):
    """Request to submit approval decision."""
    approval_id: str
    approved: bool
    scope: str = "once"  # "once", "session", "always"
    # "Approve-and-remember": on approve, add a conservative auto-approve
    # rule for this agent+repo so the Sovereign never types this again.
    remember: bool = False


class AutoModeRequest(BaseModel):
    """Request to enable or disable global Auto mode."""
    enabled: bool


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
    agent_name: Optional[str] = None
    # One-line human summary + the full command/diff preview the Sovereign
    # reviews before clicking. This is what replaces CLI typing.
    action_summary: str = ""
    command_preview: str = ""


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


class AutoModeResponse(BaseModel):
    """Global Auto mode status."""
    enabled: bool
    warning: str


# === Helper Functions ===

def get_security_feature(request: Request):
    """
    Get SecurityFeature from kestrel_sovereign.agent.

    Raises:
        HTTPException: If security feature not available
    """
    agent = get_agent(request)

    security = agent.features.get("SecurityFeature")
    if not security:
        raise HTTPException(
            status_code=503,
            detail="SecurityFeature not available"
        )

    return security


def _resolve_agent_name(security) -> Optional[str]:
    """Best-effort agent name, hardened against test mocks.

    ``getattr(mock, "_agent_name", None)`` returns a child MagicMock, not
    None, so a bare getattr would feed a non-str into the response model.
    Only a real ``str`` counts; anything else degrades to ``None``.
    """
    name = getattr(
        getattr(security.approval_queue, "_agent", None), "_agent_name", None
    )
    return name if isinstance(name, str) else None


def _command_preview(
    feature_name: str, tool_name: str, tool_args: Dict[str, Any]
) -> str:
    """The full command / diff the Sovereign reviews before deciding."""
    try:
        from kestrel_sovereign.security.auto_approve import derive_command

        cmd = derive_command(feature_name, tool_name, tool_args or {})
        if cmd:
            return cmd
    except Exception:  # noqa: BLE001
        pass
    diff = (tool_args or {}).get("diff") or (tool_args or {}).get("preview")
    if diff:
        return str(diff)
    import json as _json

    try:
        return _json.dumps(tool_args or {}, indent=2, default=str)[:4000]
    except Exception:  # noqa: BLE001
        return repr(tool_args)[:4000]


def _summarize_action(
    feature_name: str, tool_name: str, tool_args: Dict[str, Any]
) -> str:
    """One-line human summary for the row header."""
    fname = (feature_name or "").lower()
    if fname == "computer_use" and tool_name == "shell":
        argv = (tool_args or {}).get("argv") or []
        head = " ".join(str(a) for a in argv[:4])
        return f"shell: {head}".strip()
    if tool_name == "run_script":
        return (
            f"run script '{(tool_args or {}).get('script_name', '?')}' "
            f"({(tool_args or {}).get('language', '?')})"
        )
    return f"{feature_name}.{tool_name}"


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

    DENY toggles run through the demo-isolation rail (#766) — flipping
    a tool to deny on a live agent without the X-Kestrel-Allow-Destructive
    header is refused. Allow / Ask / Session are unrestricted because
    they don't disable agent capability.
    """
    security = get_security_feature(request)

    try:
        from kestrel_sovereign.features.security.permissions import PermissionLevel
        level = PermissionLevel(data.level)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid level '{data.level}'. Use: allow, auto, always_ask, deny, ask, session"
        )

    if level == PermissionLevel.DENY:
        from kestrel_sovereign.security.demo_isolation import enforce_destructive_op
        await enforce_destructive_op(request)

    if data.tool:
        await security.permission_store.set_permission(
            data.feature, data.tool, level
        )
        response = {"success": True, "message": f"Set {data.feature}.{data.tool} to {data.level}"}
    else:
        await security.permission_store.set_feature_permission(
            data.feature, level
        )
        response = {"success": True, "message": f"Set all tools in {data.feature} to {data.level}"}
    if level == PermissionLevel.AUTO:
        response["warning"] = (
            "Auto mode skips human approval when earlier constitutional, honesty, "
            "and security hooks do not flag the call. It is not a guarantee that "
            "every risk has been detected."
        )
    return response


@router.post("/permissions/feature")
@limiter.limit("30/minute")
async def set_feature_permission(request: Request, data: SetFeaturePermissionRequest):
    """
    Set permission for all tools in a feature (bulk update).

    This is the same as calling POST /permissions without a tool name.
    Bulk DENY runs through the demo-isolation rail (#766).
    """
    security = get_security_feature(request)

    try:
        from kestrel_sovereign.features.security.permissions import PermissionLevel
        level = PermissionLevel(data.level)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid level '{data.level}'. Use: allow, auto, always_ask, deny, ask, session"
        )

    if level == PermissionLevel.DENY:
        from kestrel_sovereign.security.demo_isolation import enforce_destructive_op
        await enforce_destructive_op(request)

    await security.permission_store.set_feature_permission(data.feature, level)
    response = {"success": True, "message": f"Set all tools in {data.feature} to {data.level}"}
    if level == PermissionLevel.AUTO:
        response["warning"] = (
            "Auto mode skips human approval when earlier constitutional, honesty, "
            "and security hooks do not flag the call. It is not a guarantee that "
            "every risk has been detected."
        )
    return response


@router.get("/pending", response_model=PendingListResponse)
async def get_pending_approvals(request: Request):
    """
    Get pending approval requests.

    Returns list of requests waiting for user decision.
    """
    security = get_security_feature(request)
    pending = security.approval_queue.pending_requests
    agent_name = _resolve_agent_name(security)

    return PendingListResponse(
        pending=[
            PendingApprovalResponse(
                id=r.id,
                feature=r.feature_name,
                tool=r.tool_name,
                args=r.tool_args,
                timestamp=r.created_at.isoformat(),
                agent_name=agent_name,
                action_summary=_summarize_action(
                    r.feature_name, r.tool_name, r.tool_args
                ),
                command_preview=_command_preview(
                    r.feature_name, r.tool_name, r.tool_args
                ),
            )
            for r in pending
        ],
        count=len(pending)
    )


@router.get("/auto-mode", response_model=AutoModeResponse)
async def get_auto_mode(request: Request):
    """
    Get global Auto mode status.

    Global Auto is session-scoped. While enabled, tool permissions other
    than DENY and ALWAYS_ASK behave as AUTO, so human approval is skipped
    after earlier constitutional, honesty, and security hooks do not flag.
    """
    security = get_security_feature(request)
    enabled = security.permission_store.get_global_auto_mode()
    return AutoModeResponse(
        enabled=enabled,
        warning=(
            "Global Auto skips human approval for tools that are not DENY or "
            "ALWAYS_ASK when earlier constitutional, honesty, and security "
            "hooks do not flag the call. It is session-scoped and is not a "
            "guarantee that every risk has been detected."
        ),
    )


@router.post("/auto-mode", response_model=AutoModeResponse)
@limiter.limit("30/minute")
async def set_auto_mode(request: Request, data: AutoModeRequest):
    """
    Enable or disable global Auto mode for this server session.

    Explicit DENY and ALWAYS_ASK permissions remain DENY/ALWAYS_ASK. All
    other configured or unregistered tools resolve as AUTO while this switch
    is enabled.
    """
    security = get_security_feature(request)
    security.permission_store.set_global_auto_mode(data.enabled)
    await security.permission_store.log_decision(
        feature_name="SecurityFeature",
        tool_name="global_auto_mode",
        action="mode_change",
        decision="global_auto_enabled" if data.enabled else "global_auto_disabled",
        user_choice="session",
    )
    return AutoModeResponse(
        enabled=security.permission_store.get_global_auto_mode(),
        warning=(
            "Global Auto skips human approval for tools that are not DENY or "
            "ALWAYS_ASK when earlier constitutional, honesty, and security "
            "hooks do not flag the call. It is session-scoped and is not a "
            "guarantee that every risk has been detected."
        ),
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

    # Capture the request BEFORE submit_decision resolves/pops it, so
    # "Approve-and-remember" can derive a rule from its args.
    remembered = None
    pending_req = next(
        (
            r for r in security.approval_queue.pending_requests
            if r.id == data.approval_id
        ),
        None,
    )

    decision = await security.approval_queue.submit_decision(
        data.approval_id, data.approved, data.scope
    )

    if not decision.in_memory:
        raise HTTPException(
            status_code=404,
            detail=f"Request '{data.approval_id}' not found or expired"
        )

    if data.remember and data.approved and pending_req is not None:
        try:
            from kestrel_sovereign.security.auto_approve import (
                derive_command,
                suggest_rule_from_command,
            )

            command = derive_command(
                pending_req.feature_name,
                pending_req.tool_name,
                pending_req.tool_args or {},
            )
            if command:
                pattern, repo_scope = suggest_rule_from_command(command)
                if not repo_scope:
                    # A scoped allowlist must be scoped. Refuse to remember
                    # a rule we can't bind to a repo (codex P2, #1290) —
                    # the one-off approval still stands.
                    remembered = {
                        "skipped": "no repo scope; not remembered",
                    }
                else:
                    agent_name = _resolve_agent_name(security)
                    await security.permission_store.add_auto_approve_rule(
                        pattern=pattern,
                        repo_scope=repo_scope,
                        agent=agent_name,
                        added_by="mews_approval",
                    )
                    remembered = {
                        "pattern": pattern,
                        "repo_scope": repo_scope,
                        "agent": agent_name,
                    }
        except Exception as exc:  # noqa: BLE001 - remember is best-effort
            logger.warning(
                "Approve-and-remember failed for %s: %s",
                data.approval_id, exc, exc_info=True,
            )

    result = {
        "success": True,
        "approved": data.approved,
        "scope": data.scope,
        "persisted": decision.persisted,
    }
    if decision.error is not None:
        result["warning"] = (
            "Decision was accepted in memory, but durable persistence failed"
        )
        result["persistence_error"] = decision.error
    # Only present when a rule was actually remembered.
    if remembered is not None:
        result["remembered"] = remembered
    return result


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


# === Auto-approve allowlist (Sovereign-curated, revocable) ===

@router.get("/auto-approve/rules")
async def list_auto_approve_rules(request: Request):
    """List the Sovereign-curated auto-approve rules (revocable)."""
    security = get_security_feature(request)
    rules = await security.permission_store.list_auto_approve_rules()
    return {"rules": rules, "count": len(rules)}


@router.delete("/auto-approve/rules/{rule_id}")
@limiter.limit("30/minute")
async def revoke_auto_approve_rule(request: Request, rule_id: int):
    """Revoke a remembered rule. Constitutional invariant (c):
    every auto-approve grant is revocable by removing the pattern."""
    security = get_security_feature(request)
    removed = await security.permission_store.remove_auto_approve_rule(
        rule_id
    )
    if not removed:
        raise HTTPException(
            status_code=404, detail=f"Rule {rule_id} not found"
        )
    await security.permission_store.log_decision(
        feature_name="SecurityFeature",
        tool_name="auto_approve_rule",
        action="permission_change",
        decision="auto_approve_rule_revoked",
        user_choice=str(rule_id),
    )
    return {"success": True, "revoked": rule_id}


@router.get("/auto-approve/audit")
async def get_auto_approve_audit(
    request: Request, limit: int = Query(50, ge=1, le=500)
):
    """Recent auto-approved invocations — the 'no silent runs' record
    (command, agent DID, timestamp, exit code)."""
    security = get_security_feature(request)
    rows = await security.permission_store.get_auto_approve_audit(limit)
    return {"audit": rows, "count": len(rows)}
