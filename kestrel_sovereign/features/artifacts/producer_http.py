"""Explicit finite native-output decisions under both original policies.

Only API-key/Bearer artifact callers are admitted by the canonical bridge.
No request DTO, output intent or returned plan ID confers producer authority.
"""

import json

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse, Response

from kestrel_sdk.resources import ResourceConflict
from kestrel_sovereign.artifacts.http import _delivery_error
from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.operator.runtime import OperatorRegistrationConflictError

from .producer_admission import NativeOutputLimits


async def _object(request):
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
        raise ValueError("JSON decision required")
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > 8192:
            raise ValueError("native output decision exceeds bounds")
        data.extend(chunk)
    value = json.loads(data)
    if type(value) is not dict:
        raise ValueError("native output decision object required")
    return value


def create_native_output_router(feature):
    router = APIRouter(prefix="/native-output-admissions", tags=["artifacts"])
    manager = feature.native_output_admissions

    def respond(request, *, admission_id=None, propose=False, by_run=False, action="status"):
        class DecisionResponse(Response):
            async def __call__(self, scope, receive, send):
                started = False
                source_entered = False
                async def tracked(message):
                    nonlocal started
                    if message["type"] == "http.response.start":
                        started = True
                    await send(message)
                try:
                    body = await _object(request) if request.method == "POST" else None
                    if propose or by_run:
                        if by_run:
                            body = dict(request.query_params)
                            body.pop("agent", None)
                        if (set(body) != ({"feature_name", "run_id"} if by_run else {"feature_name", "run_id", "limits"})
                                or (propose and type(body["limits"]) is not dict)):
                            raise ValueError("explicit finite native output plan required")
                        feature_name, run_id = body["feature_name"], body["run_id"]
                        if any(type(value) is not str or not value or len(value) > 256 for value in (feature_name, run_id)):
                            raise ValueError("exact native output feature and run required")
                        if propose:
                            try:
                                limits = NativeOutputLimits(**body["limits"])
                            except TypeError:
                                raise ValueError("exact finite native output limits required") from None
                        record = None
                    else:
                        record = manager._records.get(admission_id)
                        if record is None:
                            raise ResourceConflict("native output plan is unavailable")
                        feature_name, run_id = record.plan["feature_name"], record.plan["run_id"]
                    # Retain the SAME native resolver BEFORE entering artifact
                    # policy, and retain both through the actual outer send.
                    async with manager._source(feature_name, run_id, expected=record,
                            foreground=True, request=request) as source:
                        source_entered = True
                        native, identity, owner, bridge, check = source
                        async def build(original):
                            original.check_current(); check()
                            manager._check_source_scope(native, original)
                            if by_run:
                                candidates = [value for value in manager._records.values()
                                    if value.plan["feature_name"] == feature_name and value.plan["run_id"] == run_id
                                    and value.source_feature is owner and value.bridge is bridge
                                    and value.discovery_identity.matches(identity)
                                    and value.scope == (original.context.tenant_id, original.context.principal_id, original.context.acting_agent_id)]
                                value = manager._view(candidates[-1]) if candidates else {"state":"output_admission_unavailable"}
                            elif propose:
                                value = manager._propose_current(feature_name, limits, native,
                                    identity, owner, bridge, check, original)
                            else:
                                manager._check_owner(record, original)
                                if action == "decision":
                                    if (set(body) != {"plan_sha256", "allow"}
                                            or body["plan_sha256"] != record.plan_sha256):
                                        raise ResourceConflict("native output plan changed")
                                    value = await manager._decide_current(record, native, check,
                                                                          original, body["allow"])
                                elif action == "revoke":
                                    if body:
                                        raise ValueError("revocation accepts no policy fields")
                                    value = manager._revoke_current(record, original)
                                else:
                                    value = manager._view(record)
                            original.check_current(); check()
                            return JSONResponse(value)
                        await feature.artifact_response(request, build, check_current=check)(scope, receive, tracked)
                except BaseException as error:
                    if started or not isinstance(error, Exception) or is_execution_control_error(error):
                        raise
                    if not source_entered and isinstance(error,OperatorRegistrationConflictError):
                        # The existing registry refused overlapping lexical
                        # ownership. No native policy or action was admitted.
                        refused = ResourceConflict("original native policy is already in use")
                        refused.__cause__ = error
                        error = refused
                    await _delivery_error(error, False, request)(scope, receive, send)
        return DecisionResponse()

    @router.post("")
    async def propose(request: Request):
        return respond(request, propose=True)

    @router.get("")
    async def current_plan(request: Request):
        return respond(request, by_run=True)

    @router.get("/{admission_id}")
    async def status(admission_id: str, request: Request):
        return respond(request, admission_id=admission_id)

    @router.post("/{admission_id}/decision")
    async def decision(admission_id: str, request: Request):
        return respond(request, admission_id=admission_id, action="decision")

    @router.post("/{admission_id}/revoke")
    async def revoke(admission_id: str, request: Request):
        return respond(request, admission_id=admission_id, action="revoke")

    return router
