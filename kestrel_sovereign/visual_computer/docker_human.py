"""SDK0.45 human extension over exact-incarnation authenticated Docker IPC.

The normal host and worker locks require published SDK 0.45. No protocol shim
exists for older SDKs. This extension still requires the original host grant.
"""
from __future__ import annotations

import base64
import json
from pydantic import SecretBytes
from kestrel_sdk.computer_use import human_control
from kestrel_sdk.resources import ResourceUnsupported


def human_operation_id(control_id, operation, action_id=None):
    return json.dumps([control_id, operation, action_id], separators=(",", ":"))


def parse(model, value):
    from .docker import _record
    return _record(model, value)




class DockerHumanControl:
    def __init__(self, driver):
        self.driver = driver

    async def _call(self, session, control_id, operation, arguments, action_id=None):
        identifier = human_operation_id(control_id, operation, action_id)
        metadata = operation in {"status", "action_status", "revoke"}
        async def dispatch(current=None):
            return await self.driver._rpc(session.session_id, "human", {
                "session": session.model_dump(mode="json"), "operation": operation, "arguments": arguments},
                cleanup=metadata or operation == "revoke", priority=metadata or operation == "revoke", current=current)
        guard = self.driver.human_status_guard if metadata else self.driver.human_guard
        if guard is None:
            raise ResourceUnsupported("original_human_dispatch_guard_required")
        return await self.driver._guarded_call(session, identifier, guard, dispatch, workspace=True)

    async def open(self, session, request):
        return parse(human_control.HumanControlLease, await self._call(session, request.control_id,
            "open", {"request": request.model_dump(mode="json")}))

    async def status(self, session, request):
        return parse(human_control.HumanControlLease, await self._call(session, request.control_id,
            "status", {"request": request.model_dump(mode="json")}))

    async def observe(self, session, lease):
        from kestrel_sdk.computer_use import ScreenFrame, ScreenObservation
        result = await self._call(session, lease.request.control_id, "observe", {"lease": lease.model_dump(mode="json")})
        return human_control.HumanScreenObservation(control_id=result["control_id"], consumer_id=result["consumer_id"],
            observation=ScreenObservation(frame=parse(ScreenFrame, result["frame"]),
                image=SecretBytes(base64.b64decode(result["image_base64"], validate=True))))

    async def act(self, session, lease, request):
        from .dispatch import native_human_dispatch
        with native_human_dispatch(session, lease, request, self.driver.driver_id) as trace:
            try:
                return parse(human_control.HumanActionReceipt, await self._call(session, lease.request.control_id,
                    "act", {"lease": lease.model_dump(mode="json"), "request": request.model_dump(mode="json")}, request.action.action_id))
            except BaseException as error:
                fact = trace.refusal(error) if trace is not None else None
                if fact is not None:
                    raise fact from error
                raise

    async def action_status(self, session, lease, action_id):
        return parse(human_control.HumanActionReceipt, await self._call(session, lease.request.control_id,
            "action_status", {"lease": lease.model_dump(mode="json"), "action_id": action_id}, action_id))

    async def revoke(self, session, request):
        value = await self._call(session, request.control_id, "revoke", {"request": request.model_dump(mode="json")})
        model = human_control.HumanControlLease if "request_sha256" in value else human_control.HumanControlRevocationTombstone
        return parse(model, value)
