"""Candidate SDK human channel; authenticated worker IPC is its only transport.

No agent entrypoint or network authority is used. The incarnation journal stores
keyed content authentication and receipts, never pixels or typed text. A new
process refuses an existing journal rather than adopting uncertain input.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
from uuid import uuid4

from kestrel_sdk.computer_use import ComputerSession
from kestrel_sdk.computer_use.human_control import (
    HumanActionRequest, HumanControlLease, HumanControlRequest,
    HumanControlRevocation, HumanControlRevocationTombstone, HumanScreenFrame,
)
from controller import Refused, digest, instant


class HumanTransport:
    def __init__(self, controller, directory, secret):
        self.controller = controller
        self.directory = Path(directory)
        self.journal = self.directory / "human-custody.json"
        if self.journal.exists():
            raise Refused("human_conflict")
        self.key = hashlib.sha256(secret.encode() + b"human-input-v1").digest()
        self.leases, self.tombstones, self.actions, self.revocations = {}, {}, {}, {}
        self.deadlines = {}
        self.frame_at = None
        self.frame = self.focus = None
        self.epochs = set()
        self.private = False
        controller._human_action_ids = set()
        self._persist()

    def _persist(self):
        state = {"leases": self.leases, "tombstones": self.tombstones, "actions": self.actions, "revocations": self.revocations, "private": self.private}
        payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
        envelope = json.dumps({"payload": state, "authentication": hmac.new(self.key, payload, hashlib.sha256).hexdigest()}).encode()
        temporary = self.directory / ("human-" + uuid4().hex)
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(envelope)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.journal)
            descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)

    def _session(self):
        return ComputerSession.model_validate_json(json.dumps(self.controller._session))

    def _owned(self, session):
        # Metadata/revocation may target an older epoch, but never another owner,
        # resource lease, incarnation or driver. These facts grant no access.
        c = self.controller
        c._owner(session.get("scope"), session.get("session_id"))
        if any(session.get(key) != c._session[key] for key in ("lease_id", "driver_id")):
            raise Refused("human_conflict")

    def _active(self, session, lease):
        self.controller._bound(session)
        installed = self.leases.get(lease["request"]["control_id"])
        if (installed is None or HumanControlLease.model_validate_json(json.dumps(installed)) != HumanControlLease.model_validate_json(json.dumps(lease))
                or self.controller._control_pending
                or self.controller.monotonic() >= self.deadlines.get(lease["request"]["control_id"], 0)):
            raise Refused("human_conflict")
        try:
            HumanControlLease.model_validate_json(json.dumps(lease)).require_active(self._session(), at=self.controller.clock())
        except ValueError:
            raise Refused("human_conflict") from None
        return installed

    async def open(self, session, request):
        request = HumanControlRequest.model_validate_json(json.dumps(request))
        async with self.controller._lock:
            self.controller._bound(session)
            if request.control_id in self.tombstones:
                raise Refused("human_conflict")
            old = self.leases.get(request.control_id)
            if old is not None:
                try:
                    HumanControlLease.model_validate_json(json.dumps(old)).require_request(request)
                except ValueError:
                    raise Refused("human_conflict") from None
                self._active(session, old)
                self._persist()
                return copy.deepcopy(old)
            if len(self.leases) + len(self.tombstones) >= 1024 or request.control_epoch in self.epochs:
                raise Refused("human_conflict")
            try:
                request.require_session(self._session(), at=self.controller.clock())
            except ValueError:
                raise Refused("human_conflict") from None
            if (request.expires_at-request.issued_at).total_seconds() > 120 or request.max_frame_age_seconds > 15:
                raise Refused("human_unsupported")
            # Navigation is deliberately absent: manual channels cannot borrow
            # active agent action egress or mint background web authority.
            if "navigate" in request.capabilities:
                raise Refused("human_unsupported")
            lease = HumanControlLease(request=request, request_sha256=request.digest(), state="active", accepted_at=self.controller.clock())
            self.leases[request.control_id] = lease.model_dump(mode="json")
            self.deadlines[request.control_id] = self.controller.monotonic() + (request.expires_at-self.controller.clock()).total_seconds()
            self.epochs.add(request.control_epoch)
            self._persist()  # Durable binding before accepting the open.
            return copy.deepcopy(self.leases[request.control_id])

    def status(self, session, request):
        self._owned(session)
        request = HumanControlRequest.model_validate_json(json.dumps(request))
        if request.control_id in self.tombstones:
            raise Refused("human_conflict")
        lease = self.leases.get(request.control_id)
        if lease is None:
            raise Refused("human_unavailable")
        try:
            HumanControlLease.model_validate_json(json.dumps(lease)).require_request(request)
        except ValueError:
            raise Refused("human_conflict") from None
        self._persist()
        return copy.deepcopy(lease)

    async def _focus(self):
        try:
            value = await self.controller.page.evaluate("""() => { const e=document.activeElement; return [location.href,
                document.hasFocus(), e?.tagName, e?.id, e?.getAttribute('type')]; }""")
        except Exception as error:
            # No input intent exists at this point. Preserve an original known
            # refusal through act's durable denial path; cancellation still
            # belongs to the original caller and is never converted.
            raise Refused("observation_unavailable") from error
        return hmac.new(self.key, json.dumps(value).encode(), hashlib.sha256).hexdigest()

    async def observe(self, session, lease):
        async with self.controller._lock:
            current = lambda: self._active(session, lease)
            self.frame = self.focus = None
            frame, image = await self.controller._capture_locked(current)
            focus = await self._focus()
            current()
            request = lease["request"]
            frame["observation_id"] = "human-"+uuid4().hex
            record = HumanScreenFrame.model_validate_json(json.dumps({"control_id":request["control_id"],"consumer_id":request["consumer_id"],"frame":frame}))
            self.frame, self.focus, self.frame_at = record, focus, self.controller.monotonic()
            return {"control_id": record.control_id, "consumer_id": record.consumer_id,
                    "frame": frame, "image_base64": base64.b64encode(image).decode()}

    async def act(self, session, lease, request):
        request = HumanActionRequest.model_validate_json(json.dumps(request))
        async with self.controller._lock:
            # Authenticate the original issued lease separately from current
            # effect eligibility. A priority control can retire that lease
            # before this RPC enters; its exact negative fact is still owed.
            self._owned(session)
            supplied = HumanControlLease.model_validate_json(json.dumps(lease))
            installed = self.leases.get(supplied.request.control_id)
            if installed is None:
                raise Refused("human_conflict")
            original = HumanControlLease.model_validate_json(json.dumps(installed))
            try:
                original.require_request(original.request)
                supplied.require_request(original.request)
            except ValueError:
                raise Refused("human_conflict") from None
            if (supplied.accepted_at != original.accepted_at
                    or supplied.state != "active" or supplied.revoked_at is not None
                    or (request.control_id, request.consumer_id,
                        request.action.session_id, request.action.control_epoch) != (
                        original.request.control_id, original.request.consumer_id,
                        original.request.session_id, original.request.control_epoch)):
                raise Refused("human_conflict")
            action_id = request.action.action_id
            private_hash = hmac.new(self.key, request.digest().encode(), hashlib.sha256).hexdigest()
            previous = self.actions.get(action_id)
            if previous is not None:
                self._active(session, lease)
                if previous["digest"] != private_hash:
                    raise Refused("human_conflict")
                self._persist()
                return copy.deepcopy(previous["receipt"])
            if action_id in self.controller._actions:
                raise Refused("human_conflict")
            def current():
                self._active(session, lease)
                if (self.frame is None or self.frame_at is None
                        or self.controller.monotonic()-self.frame_at >= lease["request"]["max_frame_age_seconds"]):
                    raise Refused("stale_observation")
                try:
                    request.require_lease(HumanControlLease.model_validate_json(json.dumps(lease)), self._session(), self.frame, at=self.controller.clock())
                except ValueError:
                    raise Refused("stale_observation") from None
            try:
                current()
                # Revalidate the display and focus under the same lease immediately
                # before native effect. A changed page needs an explicit new view.
                actual_frame, _ = await self.controller._capture_locked(current)
                focus = await self._focus()
                current()
                if (actual_frame["image_sha256"] != self.frame.frame.image_sha256 or focus != self.focus
                        or (actual_frame["width"], actual_frame["height"]) != (self.frame.frame.width, self.frame.frame.height)):
                    raise Refused("stale_observation")
                frame = (self.frame.frame.model_dump(mode="json"), self.frame_at)
                def accepted():
                    self.controller._human_action_ids.add(action_id)
                    self.actions[action_id] = {"digest": private_hash, "receipt": self._receipt(lease, action_id)}
                    self._persist()  # Intent precedes input; no plaintext text retained.
                def persisted():
                    self.actions[action_id]["receipt"] = self._receipt(lease, action_id)
                    self._persist()
                await self.controller._input_locked(session, request.action.model_dump(mode="json"), frame,
                    private_hash, current, accepted, persisted)
                return copy.deepcopy(self.actions[action_id]["receipt"])
            except Refused as refusal:
                if self.controller._record_refusal(request.action.model_dump(mode="json"), private_hash, refusal):
                    self.controller._human_action_ids.add(action_id)
                    self.actions[action_id] = {"digest": private_hash, "receipt": self._receipt(lease, action_id)}
                    self._persist()
                raise
            finally:
                self.frame = self.focus = None

    def _receipt(self, lease, action_id):
        return {"schema_version": 1, "control_id": lease["request"]["control_id"],
                "consumer_id": lease["request"]["consumer_id"],
                "receipt": copy.deepcopy(self.controller._actions[action_id][1])}

    def action_status(self, session, lease, action_id):
        self._owned(session)
        installed = self.status(session, lease["request"])
        if installed["request_sha256"] != lease["request_sha256"]:
            raise Refused("human_conflict")
        action = self.actions.get(action_id)
        if action is None:
            raise Refused("human_unavailable")
        receipt = action["receipt"]
        if (receipt["control_id"], receipt["consumer_id"]) != (lease["request"]["control_id"], lease["request"]["consumer_id"]):
            raise Refused("human_conflict")
        self._persist()
        return copy.deepcopy(receipt)

    def fence_all(self):
        self.frame = self.focus = None
        changed = False
        for lease in self.leases.values():
            if lease["state"] == "active":
                lease["state"] = "revoking"
                changed = True
        if changed:
            self._persist()

    def finish_retirement(self):
        if any(value["receipt"]["receipt"]["outcome"] == "unknown" for value in self.actions.values()):
            raise Refused("human_outcome_unknown")
        for lease in self.leases.values():
            if lease["state"] in {"active", "revoking"}:
                lease.update(state="revoked", revoked_at=self.controller.clock().isoformat())
        self._persist()

    async def revoke(self, session, request):
        request = HumanControlRevocation.model_validate_json(json.dumps(request))
        self._owned(session)
        previous = self.revocations.get(request.control_id)
        existing = self.tombstones.get(request.control_id)
        if existing is not None:
            target = request.model_dump(mode="json", exclude={"request_id"})
            if {key:value for key,value in existing["request"].items() if key != "request_id"} != target:
                raise Refused("human_conflict")
            # A new safety request can confirm an already durable pre-open
            # fence. Preserve its original time and target; never replace an
            # accepted/unknown open with a tombstone or renew the fence.
            result = copy.deepcopy(existing)
            result["request"] = request.model_dump(mode="json")
            self._persist()
            return result
        if previous is not None and previous != request.model_dump(mode="json"):
            raise Refused("human_conflict")
        lease = self.leases.get(request.control_id)
        if lease is None:
            if len(self.leases) + len(self.tombstones) >= 1024:
                raise Refused("human_conflict")
            self.revocations[request.control_id] = request.model_dump(mode="json")
            tombstone = HumanControlRevocationTombstone(request=request, scope=self._session().scope,
                lease_id=self._session().lease_id, driver_id=self._session().driver_id, fenced_at=self.controller.clock())
            self.tombstones[request.control_id] = tombstone.model_dump(mode="json")
            self._persist()  # Atomic with open acceptance in this single loop.
            return copy.deepcopy(self.tombstones[request.control_id])
        try:
            request.require_target(HumanControlLease.model_validate_json(json.dumps(lease)), self._session())
        except ValueError:
            raise Refused("human_conflict") from None
        self.revocations[request.control_id] = request.model_dump(mode="json")
        self.fence_all()
        async with self.controller._lock:
            self.finish_retirement()
            return copy.deepcopy(lease)

    async def expire(self):
        if any(lease["state"] == "active" and (instant(lease["request"]["expires_at"]) <= self.controller.clock()
               or self.controller.monotonic() >= self.deadlines.get(lease["request"]["control_id"], 0))
               for lease in self.leases.values()):
            self.fence_all()
            async with self.controller._lock:
                self.finish_retirement()

    async def dispatch(self, session, operation, arguments):
        functions = {"open": self.open, "status": self.status, "observe": self.observe,
            "act": self.act, "action_status": self.action_status, "revoke": self.revoke}
        function = functions.get(operation)
        if function is None:
            raise Refused("human_unsupported")
        result = function(session=session, **arguments)
        return await result if hasattr(result, "__await__") else result
