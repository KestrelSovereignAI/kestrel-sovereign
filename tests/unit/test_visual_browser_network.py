"""Private active-action relay guards; real Chromium boundary is opt-in."""
import asyncio
import base64
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from tests.unit import test_visual_browser_controller as fixture

previous = sys.modules.get("controller")
sys.modules["controller"] = fixture._module
try:
    spec = importlib.util.spec_from_file_location("browser_network", Path(__file__).parents[2] / "docker/visual-browser/network.py")
    network = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(network)
finally:
    if previous is None:
        sys.modules.pop("controller", None)
    else:
        sys.modules["controller"] = previous


class Route:
    def __init__(self, *, method="GET", url="https://example.com/", body=None, headers=None):
        self.request = SimpleNamespace(method=method, url=url, post_data=body)
        async def all_headers():
            return headers or {}
        self.request.all_headers = all_headers
        self.aborted, self.fulfilled, self.gate, self.entered = False, None, None, asyncio.Event()

    async def abort(self, code):
        self.aborted = True

    async def fulfill(self, **values):
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        self.fulfilled = values


def relay():
    driver, _, session = fixture.controller()
    return network.NetworkRelay(driver), driver, session


@pytest.mark.asyncio
async def test_no_autonomous_or_wait_action_http_authority():
    value, _, session = relay()
    route = Route()
    await value.route(route)
    assert route.aborted and value.jobs == {}
    value.begin(session, {"action_id": "wait", "action": {"kind": "wait"}})
    route = Route()
    await value.route(route)
    assert route.aborted and value.jobs == {}


@pytest.mark.asyncio
async def test_job_is_exact_action_epoch_and_sealed_before_late_background_requests():
    value, _, session = relay()
    value.begin(session, {"action_id": "navigate", "action": {"kind": "navigate"}})
    route = Route()
    pending = asyncio.create_task(value.route(route)); await asyncio.sleep(0)
    jobs = value.poll(session, "navigate")
    assert len(jobs) == 1 and jobs[0]["session_id"] == session["session_id"] and jobs[0]["control_epoch"] == 0
    assert value.poll(session, "another") == [] and value.poll(session, "navigate") == []
    value.reply(session, "navigate", jobs[0]["job_id"], {"outcome": "completed", "status": 200, "headers": {},
        "body_base64": base64.b64encode(b"verified body").decode()})
    await pending
    assert await value.drain("navigate")
    assert route.fulfilled["body"] == b"verified body" and value.active is None
    background = Route()
    await value.route(background)
    assert background.aborted and not value.jobs


@pytest.mark.asyncio
async def test_cancel_drains_handlers_and_preserves_unknown_network_effect():
    value, _, session = relay()
    value.begin(session, {"action_id": "click", "action": {"kind": "click"}})
    pending = asyncio.create_task(value.route(Route())); await asyncio.sleep(0)
    draining = asyncio.create_task(value.drain("click")); await asyncio.sleep(0)
    assert (await value.cancel(session, "click"))["fenced"]
    await pending
    assert not await draining and not value.jobs and not value.handlers
    assert not await value.drain("click")


@pytest.mark.asyncio
async def test_known_method_refusal_does_not_claim_unknown_external_effect():
    value, _, session = relay()
    value.begin(session, {"action_id": "click", "action": {"kind": "click"}})
    route = Route(method="POST", body="private")
    await value.route(route)
    assert route.aborted and not value.jobs and await value.drain("click")


@pytest.mark.asyncio
@pytest.mark.parametrize("header",["cookie","Authorization","Proxy-Authorization"])
async def test_credentialed_request_is_refused_instead_of_silently_stripping_semantics(header):
    value, _, session = relay()
    value.begin(session, {"action_id":"click","action":{"kind":"click"}})
    route = Route(headers={header:"SYNTHETIC PRIVATE"})
    await value.route(route)
    assert route.aborted and not value.jobs and await value.drain("click")


@pytest.mark.asyncio
async def test_reply_and_final_delivery_recheck_current_epoch():
    value, driver, session = relay()
    value.begin(session, {"action_id":"navigate","action":{"kind":"navigate"}})
    route = Route()
    pending = asyncio.create_task(value.route(route)); await asyncio.sleep(0)
    job = value.poll(session,"navigate")[0]
    driver._session["control_epoch"] += 1
    with pytest.raises(fixture.Refused,match="stale_control"):
        value.reply(session,"navigate",job["job_id"],{"outcome":"completed","status":200,"headers":{},"body_base64":""})
    driver._session["control_epoch"] -= 1
    value.reply(session,"navigate",job["job_id"],{"outcome":"completed","status":200,"headers":{},"body_base64":""})
    driver._session["control_epoch"] += 1
    await pending
    assert route.aborted and route.fulfilled is None


@pytest.mark.asyncio
async def test_uncertain_chromium_fulfillment_retires_before_successful_takeover():
    value, driver, session = relay()
    value.begin(session, {"action_id": "click", "action": {"kind": "click"}})
    route = Route(); route.gate = asyncio.Event()
    pending = asyncio.create_task(value.route(route)); await asyncio.sleep(0)
    job = value.poll(session, "click")[0]
    value.reply(session, "click", job["job_id"], {"outcome": "completed", "status": 200, "headers": {}, "body_base64": ""})
    await route.entered.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert driver._session["state"] == "closing"
    with pytest.raises(fixture.Refused, match="stale_control"):
        await driver.control(session, fixture.control(session, "takeover"))
