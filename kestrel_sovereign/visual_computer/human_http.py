"""Private authenticated manual proxy; no agent tools or raw remote endpoint."""
from __future__ import annotations

import base64
import json
from fastapi import Request
from kestrel_sdk.resources import ResourceUnsupported
from kestrel_sovereign.security.private_delivery import AuthorizedCapabilityResponse
from .runtime import VisualAccessDenied


def add_human_routes(router, runtime, *, resolve_context):
    from .http import _body, _error, _json
    def service():
        if runtime.human is None:
            raise ResourceUnsupported("human_control_not_enabled")
        return runtime.human

    def respond(request, sid, cid, build, *, prepare=None, metadata=False, revoke=False, input=False):
        return AuthorizedCapabilityResponse(request, authority=runtime.authority,
            resolve_context=resolve_context,
            operation=lambda context,value: service().delivery(context,sid,cid,metadata=metadata,revoke=revoke,input=input),
            build=build,error_response=_error,timeout_seconds=runtime.policy.operation_seconds,
            prepare=prepare,access_denied=VisualAccessDenied,
            transport_operation=(None if revoke else lambda context,value,check:
                service().transport_delivery(context,sid,cid,check,metadata=metadata)))

    prefix = "/sessions/{session_id}/human/{control_id}"

    @router.post(prefix)
    async def open_channel(session_id: str, control_id: str, request: Request):
        async def build(context):
            return _json(await service().open(context,session_id,control_id))
        return respond(request,session_id,control_id,build)

    @router.get(prefix)
    async def status(session_id: str, control_id: str, request: Request):
        async def build(context):
            return _json(await service().status(context,session_id,control_id))
        return respond(request,session_id,control_id,build,metadata=True)

    @router.get(prefix+"/observation")
    async def observe(session_id: str, control_id: str, request: Request):
        async def build(context):
            value = await service().observe(context,session_id,control_id)
            return _json({"control_id":value.control_id,"consumer_id":value.consumer_id,
                "frame":value.observation.frame.model_dump(mode="json"),
                "image_base64":base64.b64encode(value.observation.image.get_secret_value()).decode()})
        return respond(request,session_id,control_id,build)

    @router.post(prefix+"/actions")
    async def act(session_id: str, control_id: str, request: Request):
        async def build(context,value):
            from kestrel_sdk.computer_use.human_control import HumanActionRequest
            parsed = HumanActionRequest.model_validate_json(json.dumps(value))
            return _json(await service().act(context,session_id,control_id,parsed))
        return respond(request,session_id,control_id,build,prepare=lambda:_body(request),input=True)

    @router.get(prefix+"/actions/{action_id}")
    async def action_status(session_id: str, control_id: str, action_id: str, request: Request):
        async def build(context):
            return _json(await service().action_status(context,session_id,control_id,action_id))
        return respond(request,session_id,control_id,build,metadata=True)

    @router.post(prefix+"/revoke")
    async def revoke(session_id: str, control_id: str, request: Request):
        async def build(context,value):
            if set(value) != {"request_id","control_epoch"}:
                raise ValueError("revocation requires the original request ID and epoch")
            await service().revoke(context,session_id,control_id,value["request_id"],value["control_epoch"])
            return _json({"revoked":True})
        return respond(request,session_id,control_id,build,prepare=lambda:_body(request),revoke=True)
