"""Authorized visual session routes with original private delivery custody."""

from __future__ import annotations

import base64
import json

from fastapi import APIRouter, Request
from pydantic import ValidationError
from starlette.responses import JSONResponse, Response
from kestrel_sdk.computer_use import ActionRequest, ControlRequest, EnvironmentRequest
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown, ResourceUnsupported

from kestrel_sovereign.security.private_delivery import AuthorizedCapabilityResponse
from .repository import CustodyIntegrityError
from .runtime import VisualAccessDenied, VisualAction


async def _body(request):
    if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
        raise ValueError("JSON request required")
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > 64 * 1024:
            raise ValueError("visual request exceeds byte limit")
        content.extend(chunk)
    try:
        value = json.loads(content)
    except (ValueError, UnicodeError):
        raise ValueError("invalid visual request") from None
    if not isinstance(value, dict) or "scope" in value:
        raise ValueError("visual ownership cannot be selected by input")
    return value


def _json(value, status=200):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return JSONResponse(value, status_code=status)


def _error(error, prepared):
    if prepared:
        status, code = 503, "visual_outcome_unknown"
    elif isinstance(error, OperatorAuthorizationError):
        status, code = 403, "visual_access_denied"
    elif isinstance(error, ResourceNotFound):
        status, code = 404, "visual_not_found"
    elif isinstance(error, ResourceConflict):
        status, code = 409, "visual_conflict"
    elif isinstance(error, ResourceUnsupported):
        status, code = 422, "visual_unsupported"
    elif isinstance(error, (ResourceOutcomeUnknown, TimeoutError)):
        status, code = 503, "visual_outcome_unknown"
    elif isinstance(error, CustodyIntegrityError):
        status, code = 503, "visual_custody_unavailable"
    elif isinstance(error, (ValueError, ValidationError)):
        status, code = 400, "invalid_visual_request"
    else:
        status, code = 500, "visual_service_error"
    return _json({"error":{"code":code}}, status)


def create_visual_computer_router(runtime, *, resolve_context, plan_operation=None):
    router = APIRouter(prefix="/api/visual-computer", tags=["visual-computer"])

    def respond(request, action, build, prepare=None):
        return AuthorizedCapabilityResponse(request, authority=runtime.authority,
            resolve_context=resolve_context,
            operation=lambda context,value: runtime.authority.operation(context, action, runtime.repository.storage_id),
            build=build, error_response=_error, timeout_seconds=runtime.policy.operation_seconds,
            prepare=prepare, access_denied=VisualAccessDenied)

    @router.post("/requests")
    async def admit(request: Request):
        async def build(context, value):
            value["scope"] = runtime.scope(context).model_dump(mode="json")
            admitted = EnvironmentRequest.model_validate_json(json.dumps(value))
            return _json(await runtime.admit(context, admitted))
        return respond(request, VisualAction.ADMIT, build, lambda:_body(request))

    @router.post("/requests/{request_id}/reconcile")
    async def reconcile_request(request_id: str, request: Request):
        async def build(context):
            return _json(await runtime.reconcile_request(context, request_id))
        return respond(request, VisualAction.ADMIT, build)

    @router.get("/sessions")
    async def sessions(request: Request):
        async def build(context):
            return _json(await runtime.list_sessions(context))
        return respond(request, VisualAction.STATUS, build)

    @router.get("/sessions/{session_id}")
    async def status(session_id: str, request: Request):
        async def build(context):
            return _json(await runtime.status(context, session_id))
        return respond(request, VisualAction.STATUS, build)

    @router.post("/sessions/{session_id}/reconcile")
    async def reconcile_session(session_id: str, request: Request):
        async def build(context):
            return _json(await runtime.reconcile_session(context, session_id))
        return respond(request, VisualAction.STATUS, build)

    @router.get("/sessions/{session_id}/observation")
    async def observe(session_id: str, request: Request):
        async def build(context):
            value = await runtime.observe(context, session_id)
            return _json({"frame":value.frame.model_dump(mode="json"),
                "image_base64":base64.b64encode(value.image.get_secret_value()).decode()})
        return respond(request, VisualAction.OBSERVE, build)

    @router.post("/sessions/{session_id}/actions")
    async def act(session_id: str, request: Request):
        async def build(context, value):
            action = ActionRequest.model_validate_json(json.dumps(value))
            if action.session_id != session_id:
                raise ResourceConflict("action session does not match route")
            return _json(await runtime.act(context, action))
        return respond(request, VisualAction.ACT, build, lambda:_body(request))

    @router.post("/sessions/{session_id}/plan")
    async def plan(session_id: str, request: Request):
        async def build(context, value):
            if set(value) != {"prompt"}:
                raise ValueError("planning accepts only task text")
            if plan_operation is None:
                raise ResourceUnsupported("host visual inference unavailable")
            return _json(await plan_operation(context, session_id, value["prompt"]))
        return respond(request, VisualAction.OBSERVE, build, lambda:_body(request))

    @router.get("/sessions/{session_id}/actions/{action_id}")
    async def action_status(session_id: str, action_id: str, request: Request):
        async def build(context):
            return _json(await runtime.action_status(context, session_id, action_id))
        return respond(request, VisualAction.STATUS, build)

    def control_route(operation, action):
        async def endpoint(session_id: str, request: Request):
            async def build(context, value):
                share_screen = value.pop("share_screen", False)
                if type(share_screen) is not bool:
                    raise ValueError("explicit screen sharing confirmation must be boolean")
                control = ControlRequest.model_validate_json(json.dumps(value))
                if control.operation != operation:
                    raise ResourceConflict("control operation does not match route")
                session = await runtime.control(context, session_id, control, action=action, share_screen=share_screen)
                if action == VisualAction.STOP:
                    return _json({"session_id":session_id, "stopped":session.controller == "none"})
                return _json(session)
            return respond(request, action, build, lambda:_body(request))
        return endpoint

    for suffix, operation, action in (("stop", "pause", VisualAction.STOP),
            ("takeover", "takeover", VisualAction.TAKEOVER), ("handback", "resume", VisualAction.HANDBACK)):
        router.add_api_route("/sessions/{session_id}/"+suffix, control_route(operation, action), methods=["POST"])

    @router.post("/sessions/{session_id}/release")
    async def release(session_id: str, request: Request):
        async def build(context):
            return _json(await runtime.release(context, session_id))
        return respond(request, VisualAction.RELEASE, build)

    from .human_http import add_human_routes
    add_human_routes(router, runtime, resolve_context=resolve_context)
    return router
