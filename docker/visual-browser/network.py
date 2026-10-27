"""HTTP request relay bound to one active, authenticated host action.

No browser socket can reach the host or internet: Docker network=none remains
mandatory. Requests outside the exact active action are refused locally.
"""
from __future__ import annotations

import asyncio
import base64
from uuid import uuid4
from urllib.parse import urlsplit

from controller import Refused

MAX_JOBS = 16
MAX_BODY = 128 * 1024


class NetworkRelay:
    def __init__(self, controller):
        self.controller = controller
        self.active = None
        self.jobs = {}
        self.handlers = {}
        self.last_activity = 0.0
        self.cancelled_action = None

    def begin(self, session, request):
        if self.active is not None:
            raise Refused("network_action_conflict")
        self.active = {"session": session, "action_id": request["action_id"],
                       "kind": request["action"]["kind"], "count": 0, "unknown": False}
        self.last_activity = asyncio.get_running_loop().time()

    async def route(self, route):
        active = self.active
        request = route.request
        if active is None or active["kind"] == "wait":
            await route.abort("blockedbyclient")
            return
        session = active["session"]
        identifier = uuid4().hex
        task = asyncio.current_task()
        self.handlers[identifier] = (active["action_id"], task)
        completed, refused, fulfill_started = False, False, False
        try:
            self.controller._bound(session, agent=True)
            if (urlsplit(request.url).scheme not in {"http", "https"} or request.method not in {"GET", "HEAD"}
                    or request.post_data is not None or len(request.url) > 4096 or active["count"] >= MAX_JOBS):
                refused = True
                raise Refused("network_request_refused")
            headers = await request.all_headers()
            self.controller._bound(session, agent=True)
            if any(name.lower() in {"cookie", "authorization", "proxy-authorization"} for name in headers):
                # This profile has no cookie jar or credential forwarding.
                # Refuse these semantics rather than silently stripping them.
                refused = True
                raise Refused("credentialed_http_request_unsupported")
            active["count"] += 1
            future = asyncio.get_running_loop().create_future()
            self.jobs[identifier] = {"job_id": identifier, "action_id": active["action_id"],
                "session_id": session["session_id"], "control_epoch": session["control_epoch"],
                "url": request.url, "method": request.method, "future": future, "delivered": False}
            self.last_activity = asyncio.get_running_loop().time()
            async with asyncio.timeout(9):
                reply = await future
                self.controller._bound(session, agent=True)
                if self.active is not active:
                    raise Refused("network_action_fenced")
                if reply.get("outcome") != "completed":
                    if reply.get("outcome") == "unknown":
                        active["unknown"] = True
                    else:
                        refused = True
                    raise Refused("network_request_refused")
                content = base64.b64decode(reply["body_base64"], validate=True)
                if len(content) > MAX_BODY or type(reply["status"]) is not int or not 200 <= reply["status"] <= 599:
                    raise Refused("invalid_network_response")
                # This handler remains in custody until Chromium acknowledges
                # fulfillment, so control/operation close can drain it.
                fulfill_started = True
                await route.fulfill(status=reply["status"], headers=reply["headers"], body=content)
                completed = True
        except (Refused, TimeoutError, ValueError, KeyError, TypeError):
            await route.abort("blockedbyclient")
        finally:
            if not completed and not refused:
                active["unknown"] = True
            if fulfill_started and not completed:
                # Cancelling a CDP fulfillment waiter cannot unsend response
                # delivery. Retire before reporting any successful takeover.
                self.controller._fence_retirement()
            self.jobs.pop(identifier, None)
            self.handlers.pop(identifier, None)
            self.last_activity = asyncio.get_running_loop().time()

    def poll(self, session, action_id):
        self.controller._bound(session, agent=True)
        active = self.active
        if active is None or active["action_id"] != action_id:
            return []
        result = []
        for record in self.jobs.values():
            if record["action_id"] == action_id and not record["delivered"]:
                record["delivered"] = True
                result.append({key: value for key, value in record.items() if key not in {"future", "delivered"}})
                break
        return result

    def reply(self, session, action_id, job_id, reply):
        self.controller._bound(session, agent=True)
        record = self.jobs.get(job_id)
        if (record is None or record["action_id"] != action_id or self.active is None
                or self.active["action_id"] != action_id or record["future"].done()):
            raise Refused("network_job_unavailable")
        record["future"].set_result(reply)
        return {"accepted": True}

    async def drain(self, action_id):
        # Bound subresources to this same RPC. A quiet interval admits response
        # initiated subresources; late timers/background requests are refused
        # once this action is sealed, never given autonomous host authority.
        scope = self.active
        while self.active is not None and self.active["action_id"] == action_id:
            if not self.handlers and asyncio.get_running_loop().time() - self.last_activity >= .15:
                break
            await asyncio.sleep(.02)
        active = self.active
        unknown = self.cancelled_action == action_id or bool(scope and scope["unknown"])
        if active is not None and active["action_id"] == action_id:
            self.active = None
        return not unknown

    async def cancel(self, session, action_id):
        self.controller._bound(session)
        return await self.end(action_id)

    async def end(self, action_id):
        active = self.active
        if active is not None and active["action_id"] == action_id:
            active["unknown"] = True
            self.cancelled_action = action_id
            self.active = None
        tasks = []
        for identifier, record in list(self.jobs.items()):
            if record["action_id"] == action_id and not record["future"].done():
                record["future"].set_result({"outcome": "unknown"})
        for _, (owner, task) in self.handlers.items():
            if owner == action_id:
                tasks.append(task)
        if tasks:
            async with asyncio.timeout(3):
                await asyncio.gather(*tasks, return_exceptions=True)
        return {"fenced": True}
