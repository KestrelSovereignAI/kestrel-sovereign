"""Private in-container browser state machine; the host owns admission/policy.

Wire records mirror the SDK computer contract at the host boundary. This module
has no host filesystem, provisioning, inference or artifact authority. It is
not registered as an agent capability. The authenticated transport must bind
one owner/lease before constructing a controller.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import struct
import time
from datetime import UTC, datetime
from uuid import uuid4

TOKEN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:@-]{0,254}[A-Za-z0-9])?$")
OPERATIONS = frozenset({"screenshot", "click", "type", "key", "scroll", "wait"})
KEYS = frozenset({"Enter", "Tab", "Shift+Tab", "Escape", "Backspace", "Delete", "ArrowLeft",
                  "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown",
                  "Control+A", "Shift+ArrowLeft", "Shift+ArrowRight", "Shift+Home", "Shift+End"})


class Refused(Exception):
    """A bounded public reason; never echo private input or browser diagnostics."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def identifier(value):
    if not isinstance(value, str) or not TOKEN.fullmatch(value):
        raise Refused("invalid_identifier")
    return value


def instant(value):
    try:
        result = datetime.fromisoformat(value)
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError
        return result.astimezone(UTC)
    except (ValueError, TypeError):
        raise Refused("invalid_deadline") from None


def integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def version_one(record):
    return isinstance(record, dict) and integer(record.get("schema_version"), 1, 1)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class BrowserController:
    """One page, one admitted owner, bounded receipts and monotonically fenced input.

    Reconnecting a transport may query this existing controller. Constructing a
    replacement controller is a new session, never recovery of accepted actions.
    Frames identify the latest produced observation, not a semantic guarantee
    that autonomous page content cannot change. The host governs consequential
    effects; this worker does not infer permission from a model's click intent.
    """

    def __init__(self, page, request: dict, *, driver_id="local-docker-browser",
                 clock=lambda: datetime.now(UTC), monotonic=time.monotonic,
                 action_limit=4096, frame_ttl_seconds=30, workspace_enabled=False, network_enabled=False):
        self.operations = OPERATIONS | ({"files"} if workspace_enabled else set()) | ({"navigate"} if network_enabled else set())
        required = {"schema_version", "session_id", "scope", "lease_id", "surface",
                    "required_capabilities", "expires_at"}
        if not version_one(request) or set(request) != required:
            raise Refused("invalid_session")
        scope = request["scope"]
        if not version_one(scope) or set(scope) != {"schema_version", "tenant_id", "principal_id", "agent_id"}:
            raise Refused("invalid_scope")
        for field in ("tenant_id", "principal_id"):
            identifier(scope[field])
        if scope["agent_id"] is not None:
            identifier(scope["agent_id"])
        capabilities = request["required_capabilities"]
        if (request["surface"] != "browser" or not isinstance(capabilities, list)
                or not all(isinstance(item, str) for item in capabilities)
                or "screenshot" not in capabilities or not set(capabilities) <= self.operations):
            raise Refused("unsupported_capability")
        self.clock, self.monotonic = clock, monotonic
        remaining = (instant(request["expires_at"]) - clock()).total_seconds()
        if not 0 < remaining <= 3600:
            raise Refused("invalid_session_lifetime")
        if not integer(action_limit, 1, 4096) or not integer(frame_ttl_seconds, 1, 30):
            raise Refused("invalid_limits")
        self.deadline = monotonic() + remaining
        self.page, self.action_limit, self.frame_ttl = page, action_limit, frame_ttl_seconds
        self._session = {"schema_version": 1, "session_id": identifier(request["session_id"]),
                         "scope": copy.deepcopy(scope), "lease_id": identifier(request["lease_id"]),
                         "driver_id": identifier(driver_id), "surface": "browser",
                         "capabilities": sorted(self.operations | {"takeover"}), "state": "ready", "controller": "agent",
                         "control_epoch": 0, "expires_at": request["expires_at"]}
        self._lock = asyncio.Lock()
        self._frame = None
        self._actions = {}
        self._controls = {}
        self._control_pending = False

    def _live(self):
        if self.monotonic() >= self.deadline or self.clock() >= instant(self._session["expires_at"]):
            self._fence_retirement()
            raise Refused("session_expired")

    def _fence_retirement(self):
        if self._session["state"] not in {"closing", "closed"}:
            self._session.update(state="closing", controller="none", control_epoch=self._session["control_epoch"] + 1)
        self._frame = None

    def _owner(self, scope, session_id):
        if not version_one(scope) or scope != self._session["scope"] or session_id != self._session["session_id"]:
            raise Refused("session_unavailable")

    def _bound(self, session, *, agent=False):
        if not version_one(session) or not version_one(session.get("scope")):
            raise Refused("session_unavailable")
        self._owner(session.get("scope"), session.get("session_id"))
        if (session.get("lease_id") != self._session["lease_id"]
                or session.get("driver_id") != self._session["driver_id"]
                or type(session.get("control_epoch")) is not int
                or session["control_epoch"] != self._session["control_epoch"]):
            raise Refused("stale_control")
        self._live()
        if agent and (self._session["state"] != "ready" or self._control_pending):
            raise Refused("input_fenced")

    def status(self, scope, session_id):
        self._owner(scope, session_id)
        try:
            self._live()
        except Refused:
            pass  # Expired custody remains inspectable, never falsely ready.
        return copy.deepcopy(self._session)

    async def retire(self):
        """Worker-owned expiry/disconnect cleanup; no agent command calls this."""
        if self._session["state"] == "closed":
            return True
        self._fence_retirement()
        try:
            async with asyncio.timeout(5):
                await self.page.context.close()
        except Exception:  # noqa: BLE001 — boundary keeps private browser details out of IPC/logs
            return False
        self._session["state"] = "closed"
        return True

    async def observe(self, session):
        async with self._lock:
            self._bound(session, agent=True)
            try:
                async with asyncio.timeout(min(10, self.deadline - self.monotonic())):
                    image = await self.page.screenshot(type="png", caret="hide")
                self._bound(session, agent=True)
                if not 24 <= len(image) <= 16 * 1024 * 1024 or image[:8] != b"\x89PNG\r\n\x1a\n":
                    raise Refused("invalid_observation")
                width, height = struct.unpack(">II", image[16:24])
                if not 1 <= width <= 16384 or not 1 <= height <= 16384 or width * height > 16_777_216:
                    raise Refused("invalid_observation")
                frame = {"schema_version": 1, "observation_id": str(uuid4()), "session_id": self._session["session_id"],
                         "control_epoch": self._session["control_epoch"], "observed_at": self.clock().isoformat(),
                         "width": width, "height": height, "image_sha256": hashlib.sha256(image).hexdigest(), "media_type": "image/png"}
                self._frame = (frame, self.monotonic())
                return copy.deepcopy(frame), image
            except Refused:
                raise
            except Exception:  # noqa: BLE001 — boundary keeps private browser details out of IPC/logs
                self._frame = None
                raise Refused("observation_unavailable") from None

    def _effect(self, action, frame):
        if not version_one(action):
            raise Refused("invalid_action")
        kind = action.get("kind")
        if (kind == "click" and set(action) == {"schema_version", "kind", "x", "y", "button"}
                and integer(action["x"], 0, frame["width"] - 1) and integer(action["y"], 0, frame["height"] - 1)
                and isinstance(action["button"], str) and action["button"] in {"left", "middle", "right"}):
            return lambda: self.page.mouse.click(action["x"], action["y"], button=action["button"])
        if (kind == "type" and set(action) == {"schema_version", "kind", "text"}
                and isinstance(action["text"], str) and 1 <= len(action["text"]) <= 16384):
            return lambda: self.page.keyboard.insert_text(action["text"])
        if (kind == "key" and set(action) == {"schema_version", "kind", "key"}
                and isinstance(action["key"], str) and action["key"] in KEYS):
            return lambda: self.page.keyboard.press(action["key"])
        if (kind == "scroll" and set(action) == {"schema_version", "kind", "delta_x", "delta_y"}
                and all(integer(action[name], -10000, 10000) for name in ("delta_x", "delta_y"))):
            return lambda: self.page.mouse.wheel(action["delta_x"], action["delta_y"])
        if (kind == "navigate" and "navigate" in self.operations
                and set(action) == {"schema_version", "kind", "url"}):
            from kestrel_sdk.computer_use import Navigate
            from pydantic import ValidationError
            try:
                navigation = Navigate.model_validate_json(json.dumps(action))
            except (ValidationError, ValueError, TypeError):
                raise Refused("unsupported_action") from None
            return lambda: self.page.goto(navigation.url, wait_until="domcontentloaded")
        if (kind == "wait" and set(action) == {"schema_version", "kind", "milliseconds"}
                and integer(action["milliseconds"], 0, 10000)):
            return lambda: asyncio.sleep(action["milliseconds"] / 1000)
        raise Refused("unsupported_action")

    async def act(self, session, request):
        async with self._lock:
            self._bound(session, agent=True)
            fields = {"schema_version", "action_id", "session_id", "observation_id", "control_epoch", "not_after", "action"}
            if not version_one(request) or set(request) != fields:
                raise Refused("invalid_action")
            action_id = identifier(request["action_id"])
            if (request["session_id"] != self._session["session_id"] or type(request["control_epoch"]) is not int
                    or request["control_epoch"] != self._session["control_epoch"]):
                raise Refused("stale_control")
            request_hash = digest(request)
            previous = self._actions.get(action_id)
            if previous:
                if previous[0] != request_hash:
                    raise Refused("action_conflict")
                return copy.deepcopy(previous[1])
            if len(self._actions) >= self.action_limit:
                raise Refused("action_budget_exhausted")
            if (self._frame is None or request["observation_id"] != self._frame[0]["observation_id"]
                    or self.monotonic() - self._frame[1] > self.frame_ttl):
                raise Refused("stale_observation")
            not_after = instant(request["not_after"])
            effect = self._effect(request["action"], self._frame[0])
            is_wait = request["action"]["kind"] == "wait"
            frame_deadline = self._frame[1] + (not_after - instant(self._frame[0]["observed_at"])).total_seconds()
            remaining = min((not_after - self.clock()).total_seconds(), frame_deadline - self.monotonic(),
                            self.deadline - self.monotonic(), 11 if is_wait else 10)
            if remaining <= 0 or (is_wait and request["action"]["milliseconds"] / 1000 >= remaining):
                raise Refused("action_expired")
            receipt = {"schema_version": 1, "action_id": action_id, "session_id": self._session["session_id"],
                       "observation_id": request["observation_id"], "control_epoch": request["control_epoch"],
                       "outcome": "unknown", "recorded_at": self.clock().isoformat(), "reason_code": "outcome_unknown"}
            self._actions[action_id] = (request_hash, receipt)
            self._frame = None
            relay = getattr(self, "network", None)
            if relay is not None:
                relay.begin(session, request)
            try:
                # No await separates final input fencing from effect dispatch.
                self._bound(session, agent=True)
                async with asyncio.timeout(remaining):
                    await effect()
                    network_known = await relay.drain(action_id) if relay is not None else True
                receipt.update(outcome="completed" if network_known else "unknown",
                               reason_code=None if network_known else "network_outcome_unknown", recorded_at=self.clock().isoformat())
            except asyncio.CancelledError:
                # Cancelling the Python waiter does not unsend a Chromium input.
                # Fence synchronously; the worker must physically retire it.
                if is_wait:
                    receipt.update(outcome="denied", reason_code="wait_cancelled", recorded_at=self.clock().isoformat())
                else:
                    self._fence_retirement()
                raise
            except Exception:  # noqa: BLE001 — accepted effect remains explicitly unknown
                # A renderer can deliver a timed-out click later. Do not report
                # a successful pause or permit resume with that input pending.
                if is_wait:
                    # Wait never sends browser input, so interruption is known
                    # to have no pending physical effect and needs no retirement.
                    receipt.update(outcome="denied", reason_code="wait_interrupted", recorded_at=self.clock().isoformat())
                else:
                    self._fence_retirement()
                    await self.retire()
            finally:
                if relay is not None:
                    try:
                        await relay.end(action_id)
                    except (TimeoutError, ConnectionError, OSError):
                        self._fence_retirement()
                        receipt.update(outcome="unknown", reason_code="network_outcome_unknown")
            return copy.deepcopy(receipt)

    def action_status(self, session, action_id):
        self._bound(session)
        previous = self._actions.get(identifier(action_id))
        if previous is None:
            raise Refused("action_unavailable")
        return copy.deepcopy(previous[1])

    async def control(self, session, request):
        self._bound(session)
        fields = {"schema_version", "request_id", "session_id", "expected_control_epoch", "operation"}
        if not version_one(request) or set(request) != fields:
            raise Refused("invalid_control")
        request_id = identifier(request["request_id"])
        if (request["session_id"] != self._session["session_id"] or type(request["expected_control_epoch"]) is not int
                or request["expected_control_epoch"] != self._session["control_epoch"]):
            raise Refused("stale_control")
        if not isinstance(request["operation"], str) or request["operation"] not in {"pause", "takeover", "resume", "close"}:
            raise Refused("unsupported_control")
        if self._control_pending:
            raise Refused("control_pending")
        if len(self._controls) >= 1024 and request["operation"] == "resume":
            raise Refused("control_budget_exhausted")
        if request_id in self._controls:
            raise Refused("control_conflict")
        self._control_pending = True
        try:
            async with self._lock:
                self._bound(session)
                if self._session["state"] == "closed":
                    raise Refused("session_closed")
                operation = request["operation"]
                if ((operation == "resume" and self._session["state"] != "paused")
                        or (operation == "pause" and self._session["state"] != "ready")
                        or (operation == "takeover" and self._session["state"] not in {"ready", "paused"})):
                    raise Refused("invalid_control_transition")
                self._session.update(control_epoch=self._session["control_epoch"] + 1,
                                     state="ready" if operation == "resume" else "paused" if operation in {"pause", "takeover"} else "closing",
                                     controller="agent" if operation == "resume" else "human" if operation == "takeover" else "none")
                self._frame = None
                if len(self._controls) < 1024:
                    self._controls[request_id] = digest(request)
                if operation == "close":
                    try:
                        async with asyncio.timeout(5):
                            await self.page.context.close()
                    except Exception:  # noqa: BLE001 — boundary keeps private browser details out of IPC/logs
                        raise Refused("control_outcome_unknown") from None
                    self._session["state"] = "closed"
                return copy.deepcopy(self._session)
        finally:
            self._control_pending = False
