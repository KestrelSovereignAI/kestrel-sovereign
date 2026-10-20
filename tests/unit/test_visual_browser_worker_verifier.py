"""Reject misleading real-worker acceptance reports before claiming expiry."""
import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from scripts.verify_visual_browser_worker import exercise
from tests.unit.test_visual_browser_controller import Page
from tests.unit.test_visual_browser_worker import worker


@pytest.mark.parametrize("exit_code,after_expiry,oom,wait_elapsed,valid", [
    (0, False, False, 10, False), (1, False, False, 10, False), (1, True, False, 10, False),
    (0, True, True, 10, False), (0, True, False, 10, True), (0, True, False, 0, False),
])
def test_worker_verifier_requires_normal_physical_expiry(tmp_path, exit_code, after_expiry, oom, wait_elapsed, valid):
    endpoint = None
    elapsed = [0]
    page = Page()
    original_screenshot = page.screenshot

    async def screenshot(**options):
        return await original_screenshot(**options) + str(len(page.effects)).encode()

    page.screenshot = screenshot

    def checked(arguments, *, input=None, **options):
        nonlocal endpoint
        if arguments[-1] == "initialize":
            initial = json.loads(input)
            endpoint = worker.Endpoint(worker.BrowserController(page, initial["request"]), initial["secret"])
            # These cases test expiry evidence, not wait timing (covered by the
            # controller suite and real Docker). Keep their fake worker instant.
            effect = endpoint.controller._effect
            endpoint.controller._effect = lambda action, frame: ((lambda: asyncio.sleep(0))
                if action["kind"] == "wait" else effect(action, frame))
            return b'{"submitted":true}'
        if arguments[-1] == "/opt/kestrel-browser/rpc_client.py":
            request = json.loads(input)
            try:
                response = {"result": asyncio.run(endpoint.dispatch(request))}
            except worker.Refused as error:
                response = {"error": error.code}
            if request["operation"] == "act" and request["arguments"]["request"]["action"]["kind"] == "wait":
                elapsed[0] += wait_elapsed
            return json.dumps({"schema_version": 1, "id": request["id"], **response}).encode()
        if arguments[:2] == ["container", "inspect"]:
            finished = (datetime.fromisoformat(endpoint.controller._session["expires_at"]) + timedelta(seconds=1)
                        if after_expiry else datetime.now(UTC))
            return json.dumps([{"State": {"Running": False, "ExitCode": exit_code,
                                           "FinishedAt": finished.isoformat(), "OOMKilled": oom}}]).encode()
        if "admission.json" in arguments[-1]:
            return b'{"sandbox":"synthetic proof","browser_version":"synthetic"}'
        return b""

    if valid:
        result = exercise(checked, "synthetic-container", tmp_path, monotonic=lambda: elapsed[0])
        assert result["expiry_physically_retired"] is True and result["retired_exit_code"] == 0
        assert datetime.fromisoformat(result["finished_at"]) >= datetime.fromisoformat(result["expires_at"])
    else:
        with pytest.raises(RuntimeError, match="expiry|maximum wait"):
            exercise(checked, "synthetic-container", tmp_path, monotonic=lambda: elapsed[0])
