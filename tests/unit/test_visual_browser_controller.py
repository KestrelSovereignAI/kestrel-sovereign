"""Private browser worker execution boundaries; no host capability is enabled."""
import asyncio
import importlib.util
import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("visual_browser_controller", Path(__file__).parents[2] / "docker/visual-browser/controller.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
BrowserController, Refused = _module.BrowserController, _module.Refused


class Page:
    def __init__(self):
        self.mouse = self.keyboard = self.context = self
        self.effects = []
        self.gate = None
        self.entered = asyncio.Event()
        self.closed = False
        self.fail_close = False

    async def screenshot(self, **options):
        return b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", 800, 600) + b"synthetic pixels"

    async def click(self, *args, **kwargs):
        self.effects.append((args, kwargs))
        self.entered.set()
        if self.gate:
            await self.gate.wait()

    insert_text = press = wheel = click

    async def close(self):
        if self.fail_close:
            raise RuntimeError("private browser diagnostic")
        self.closed = True


def controller(**options):
    now = datetime.now(UTC)
    scope = {"schema_version": 1, "tenant_id": "tenant", "principal_id": "owner", "agent_id": "agent"}
    request = {"schema_version": 1, "session_id": "session", "lease_id": "lease", "scope": scope,
               "surface": "browser", "required_capabilities": ["screenshot"], "expires_at": (now + timedelta(minutes=5)).isoformat()}
    page = Page()
    driver = BrowserController(page, request, **options)
    return driver, page, driver.status(scope, "session")


async def action(driver, session, *, action_id="input", effect=None):
    frame, image = await driver.observe(session)
    assert image.startswith(b"\x89PNG")
    return {"schema_version": 1, "action_id": action_id, "session_id": session["session_id"],
            "observation_id": frame["observation_id"], "control_epoch": session["control_epoch"],
            "not_after": (datetime.now(UTC) + timedelta(seconds=5)).isoformat(),
            "action": effect or {"schema_version": 1, "kind": "click", "x": 1, "y": 2, "button": "left"}}


def control(session, operation, request_id="control"):
    return {"schema_version": 1, "request_id": request_id, "session_id": session["session_id"],
            "expected_control_epoch": session["control_epoch"], "operation": operation}


@pytest.mark.asyncio
async def test_accepted_input_is_not_replayed_and_different_reuse_conflicts():
    driver, page, session = controller()
    request = await action(driver, session)
    receipt = await driver.act(session, request)
    assert receipt["outcome"] == "completed"
    assert await driver.act(session, request) == receipt
    assert driver.action_status(session, "input") == receipt
    assert len(page.effects) == 1
    with pytest.raises(Refused, match="action_conflict"):
        await driver.act(session, {**request, "action": {**request["action"], "x": 4}})
    assert len(page.effects) == 1


@pytest.mark.asyncio
async def test_cancelled_accepted_effect_remains_unknown_without_replay():
    driver, page, session = controller()
    request = await action(driver, session)
    page.gate = asyncio.Event()
    task = asyncio.create_task(driver.act(session, request))
    await page.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    closing = driver.status(session["scope"], session["session_id"])
    assert closing["state"] == "closing" and closing["control_epoch"] == 1
    assert driver.action_status(closing, "input")["outcome"] == "unknown"
    with pytest.raises(Refused, match="stale_control"):
        await driver.act(session, request)
    assert await driver.retire()
    assert len(page.effects) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["pause", "takeover"])
async def test_pending_pause_fences_queued_input_and_drains_accepted_effect(operation):
    driver, page, session = controller()
    request = await action(driver, session)
    page.gate = asyncio.Event()
    active = asyncio.create_task(driver.act(session, request))
    await page.entered.wait()
    queued = asyncio.create_task(driver.act(session, {**request, "action_id": "queued"}))
    pause = asyncio.create_task(driver.control(session, control(session, operation)))
    await asyncio.sleep(0)
    page.gate.set()
    assert (await active)["outcome"] == "completed"
    with pytest.raises(Refused, match="input_fenced"):
        await queued
    paused = await pause
    assert paused["state"] == "paused" and paused["control_epoch"] == 1
    assert paused["controller"] == ("human" if operation == "takeover" else "none")
    assert len(page.effects) == 1
    with pytest.raises(Refused, match="stale_control"):
        await driver.observe(session)
    with pytest.raises(Refused, match="input_fenced"):
        await driver.observe(paused)
    resumed = await driver.control(paused, control(paused, "resume", "resume"))
    assert resumed["control_epoch"] == 2
    with pytest.raises(Refused, match="stale_control"):
        await driver.act(resumed, request)


@pytest.mark.asyncio
async def test_close_uncertainty_cannot_be_resumed_or_claimed_closed():
    driver, page, session = controller()
    page.fail_close = True
    with pytest.raises(Refused, match="control_outcome_unknown") as error:
        await driver.control(session, control(session, "close"))
    assert "private" not in str(error.value)
    closing = driver.status(session["scope"], session["session_id"])
    assert closing["state"] == "closing"
    with pytest.raises(Refused, match="invalid_control_transition"):
        await driver.control(closing, control(closing, "resume", "resume"))
    page.fail_close = False
    closed = await driver.control(closing, control(closing, "close", "retry-close"))
    assert closed["state"] == "closed" and page.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("change,reason", [
    ({"observation_id": "older-frame"}, "stale_observation"),
    ({"control_epoch": 1}, "stale_control"),
    ({"session_id": "other"}, "stale_control"),
    ({"not_after": "2020-01-01T00:00:00+00:00"}, "action_expired"),
    ({"action": {"schema_version": 1, "kind": "click", "x": 800, "y": 1, "button": "left"}}, "unsupported_action"),
    ({"action": {"schema_version": 1, "kind": "click", "x": True, "y": 1, "button": "left"}}, "unsupported_action"),
    ({"action": {"schema_version": 1, "kind": "key", "key": "Control+V"}}, "unsupported_action"),
    ({"action": {"schema_version": 1, "kind": "navigate", "url": "http://localhost"}}, "unsupported_action"),
])
async def test_stale_expired_or_unsupported_input_has_no_effect(change, reason):
    driver, page, session = controller()
    request = await action(driver, session)
    with pytest.raises(Refused, match=reason):
        await driver.act(session, {**request, **change})
    assert page.effects == []


@pytest.mark.asyncio
async def test_owner_epoch_frame_age_and_receipt_budget_are_enforced():
    elapsed = [0]
    driver, page, session = controller(monotonic=lambda: elapsed[0], action_limit=1)
    request = await action(driver, session)
    elapsed[0] = 31
    with pytest.raises(Refused, match="stale_observation"):
        await driver.act(session, request)
    request = await action(driver, session)
    await driver.act(session, request)
    next_request = await action(driver, session, action_id="next")
    with pytest.raises(Refused, match="action_budget_exhausted"):
        await driver.act(session, next_request)
    wrong = {**session, "scope": {**session["scope"], "principal_id": "other"}}
    with pytest.raises(Refused, match="session_unavailable"):
        await driver.observe(wrong)
    session["scope"]["principal_id"] = "mutated-return-value"
    with pytest.raises(Refused, match="session_unavailable"):
        driver.status(session["scope"], session["session_id"])
    assert len(page.effects) == 1


@pytest.mark.asyncio
async def test_boolean_versions_and_clock_rollback_cannot_extend_input_authority():
    now = [datetime.now(UTC)]
    elapsed = [0]
    driver, page, session = controller(clock=lambda: now[0], monotonic=lambda: elapsed[0])
    request = await action(driver, session)
    with pytest.raises(Refused, match="session_unavailable"):
        driver.status({**session["scope"], "schema_version": True}, session["session_id"])
    with pytest.raises(Refused, match="session_unavailable"):
        await driver.observe({**session, "schema_version": True})
    with pytest.raises(Refused, match="invalid_action"):
        await driver.act(session, {**request, "schema_version": True})
    with pytest.raises(Refused, match="invalid_action"):
        await driver.act(session, {**request, "action": {**request["action"], "schema_version": True}})
    elapsed[0] = 6
    now[0] -= timedelta(hours=1)
    with pytest.raises(Refused, match="action_expired"):
        await driver.act(session, request)
    assert page.effects == []
    elapsed[0] = 301
    assert driver.status(session["scope"], session["session_id"])["state"] == "closing"
    assert not page.closed
    assert await driver.retire()
    assert page.closed
    assert driver.status(session["scope"], session["session_id"])["state"] == "closed"


@pytest.mark.asyncio
async def test_exhausting_control_budget_preserves_pause_and_close():
    driver, page, session = controller()
    for index in range(1024):
        session = await driver.control(session, control(session, "pause" if index % 2 == 0 else "resume", f"control-{index}"))
    assert session["state"] == "ready"
    paused = await driver.control(session, control(session, "pause", "safety-pause"))
    assert paused["state"] == "paused"
    with pytest.raises(Refused, match="control_budget_exhausted"):
        await driver.control(paused, control(paused, "resume", "exhausted-resume"))
    closed = await driver.control(paused, control(paused, "close", "safety-close"))
    assert closed["state"] == "closed" and page.closed
    assert len(driver._controls) <= 1024


@pytest.mark.asyncio
async def test_timed_out_input_retires_instead_of_reporting_a_quiescent_pause():
    driver, page, session = controller()
    request = await action(driver, session)
    async def timeout_after_dispatch(*args, **kwargs):
        page.effects.append((args, kwargs))
        raise TimeoutError
    page.click = timeout_after_dispatch
    assert (await driver.act(session, request))["outcome"] == "unknown"
    current = driver.status(session["scope"], session["session_id"])
    assert current["state"] == "closed" and page.closed
    assert driver.action_status(current, "input")["outcome"] == "unknown"
    with pytest.raises(Refused, match="stale_control"):
        await driver.control(session, control(session, "pause"))
    with pytest.raises(Refused, match="session_closed"):
        await driver.control(current, control(current, "resume"))
    assert len(page.effects) == 1


@pytest.mark.asyncio
async def test_maximum_wait_completes_without_retiring_the_persistent_page():
    driver, page, session = controller()
    request = await action(driver, session, effect={"schema_version": 1, "kind": "wait", "milliseconds": 10000})
    request["not_after"] = (datetime.now(UTC) + timedelta(seconds=60)).isoformat()
    started = asyncio.get_running_loop().time()
    receipt = await driver.act(session, request)
    assert asyncio.get_running_loop().time() - started >= 9.9
    assert receipt["outcome"] == "completed"
    assert driver.status(session["scope"], session["session_id"])["state"] == "ready"
    assert not page.closed and page.effects == []


@pytest.mark.asyncio
async def test_wait_that_cannot_finish_by_deadline_is_refused_without_retirement():
    driver, page, session = controller()
    request = await action(driver, session, effect={"schema_version": 1, "kind": "wait", "milliseconds": 300})
    request["not_after"] = (datetime.now(UTC) + timedelta(milliseconds=200)).isoformat()
    with pytest.raises(Refused, match="action_expired"):
        await driver.act(session, request)
    assert driver.status(session["scope"], session["session_id"])["state"] == "ready"
    assert not page.closed and page.effects == []


@pytest.mark.asyncio
@pytest.mark.parametrize("effect", [
    {"schema_version": 1, "kind": "key", "key": "ArrowDown"},
    {"schema_version": 1, "kind": "scroll", "delta_x": 0, "delta_y": 10},
])
async def test_supported_key_and_scroll_complete_in_the_same_session(effect):
    driver, page, session = controller()
    request = await action(driver, session, effect=effect)
    assert (await driver.act(session, request))["outcome"] == "completed"
    assert len(page.effects) == 1 and not page.closed


@pytest.mark.asyncio
async def test_cancelled_wait_has_a_known_receipt_and_keeps_the_page():
    driver, page, session = controller()
    request = await action(driver, session, effect={"schema_version": 1, "kind": "wait", "milliseconds": 1000})
    task = asyncio.create_task(driver.act(session, request))
    await asyncio.sleep(0)
    assert "input" in driver._actions
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert driver.action_status(session, "input")["outcome"] == "denied"
    assert driver.status(session["scope"], session["session_id"])["state"] == "ready"
    assert page.effects == [] and not page.closed


@pytest.mark.asyncio
async def test_interrupted_wait_does_not_enter_uncertain_browser_input_retirement():
    driver, page, session = controller()
    request = await action(driver, session, effect={"schema_version": 1, "kind": "wait", "milliseconds": 1})
    async def interrupted_sleep():
        raise TimeoutError
    driver._effect = lambda action, frame: interrupted_sleep
    receipt = await driver.act(session, request)
    assert receipt["outcome"] == "denied" and receipt["reason_code"] == "wait_interrupted"
    assert driver.status(session["scope"], session["session_id"])["state"] == "ready"
    assert page.effects == [] and not page.closed
