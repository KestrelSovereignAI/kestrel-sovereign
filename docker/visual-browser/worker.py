"""Private offline browser worker, reached only through authenticated Unix IPC.

The host must admit the exact image/container configuration before starting this
process. It supplies one session and a fresh secret over Docker exec stdin.
There is no public listener, CDP endpoint, filesystem transfer or provider API.
"""
import asyncio
import base64
import hmac
import math
import os
import signal
import secrets
from datetime import UTC, datetime

from controller import BrowserController, Refused, identifier, instant, version_one
from sandbox_checks import chromium_evidence, process_evidence
from wire import DIRECTORY, MAX_REQUEST, MAX_RESPONSE, SECRET, decode, encode


def bootstrap(value):
    if (not version_one(value) or set(value) != {"schema_version", "secret", "request", "document_html", "incarnation"}
            or not isinstance(value["secret"], str) or not SECRET.fullmatch(value["secret"])
            or not isinstance(value["incarnation"], str) or not SECRET.fullmatch(value["incarnation"])
            or not isinstance(value["document_html"], str)
            or len(value["document_html"].encode()) > 65536):
        raise Refused("invalid_bootstrap")
    return value


class Endpoint:
    """Transport retries query the same controller; they never construct a new one."""

    def __init__(self, controller, secret, *, priority=False):
        self.controller, self.secret = controller, secret
        self.connections = 0
        self.priority = priority

    async def dispatch(self, message):
        if (not version_one(message) or set(message) != {"schema_version", "secret", "id", "operation", "arguments"}
                or not isinstance(message["secret"], str) or not SECRET.fullmatch(message["secret"])
                or not hmac.compare_digest(message["secret"], self.secret)):
            raise Refused("unauthenticated")
        identifier(message["id"])
        arguments, operation = message["arguments"], message["operation"]
        if not isinstance(arguments, dict):
            raise Refused("invalid_request")
        controller = self.controller
        if self.priority and operation not in {"status", "action_status", "control"}:
            raise Refused("priority_operation_required")
        if operation == "status" and set(arguments) == {"scope", "session_id"}:
            return controller.status(**arguments)
        if operation == "observe" and set(arguments) == {"session"}:
            frame, data = await controller.observe(**arguments)
            return {"frame": frame, "image_base64": base64.b64encode(data).decode()}
        if operation == "act" and set(arguments) == {"session", "request"}:
            return await controller.act(**arguments)
        if operation == "action_status" and set(arguments) == {"session", "action_id"}:
            return controller.action_status(**arguments)
        if operation == "control" and set(arguments) == {"session", "request"}:
            return await controller.control(**arguments)
        if operation == "workspace" and set(arguments) == {"session", "operation", "arguments"} and hasattr(controller, "workspace"):
            return await controller.workspace.dispatch(**arguments)
        raise Refused("unsupported_request")

    async def handle(self, reader, writer):
        # Do not queue an unbounded number of requests behind the controller lock.
        if self.connections >= (2 if self.priority else 4):
            writer.close()
            return
        self.connections += 1
        try:
            async with asyncio.timeout(20):
                request_id = None
                try:
                    async with asyncio.timeout(1 if self.priority else 20):
                        raw = await reader.readline()
                    if not raw.endswith(b"\n") or len(raw) > MAX_REQUEST:
                        raise Refused("invalid_request")
                    message = decode(raw)
                    if isinstance(message, dict):
                        try:
                            request_id = identifier(message.get("id"))
                        except Refused:
                            pass
                    result = {"schema_version": 1, "id": request_id, "result": await self.dispatch(message)}
                except Refused as error:
                    result = {"schema_version": 1, "id": request_id, "error": error.code}
                except (ValueError, TypeError, KeyError, RecursionError, UnicodeError):
                    result = {"schema_version": 1, "id": request_id, "error": "invalid_request"}
                response = encode(result)
                if len(response) > MAX_RESPONSE:
                    raise Refused("response_too_large")
                writer.write(response)
                await writer.drain()
        except (TimeoutError, ConnectionError, Refused, OSError):
            # A lost response is not evidence that an accepted action failed.
            pass
        finally:
            self.connections -= 1
            writer.close()
            try:
                async with asyncio.timeout(1):
                    await writer.wait_closed()
            except (TimeoutError, ConnectionError, OSError):
                pass


async def serve(controller, secret, *, directory=DIRECTORY):
    endpoint = Endpoint(controller, secret)
    priority = Endpoint(controller, secret, priority=True)
    socket_path = directory / "worker.sock"
    priority_path = directory / "control.sock"
    server = await asyncio.start_unix_server(endpoint.handle, str(socket_path), limit=MAX_REQUEST)
    control_server = await asyncio.start_unix_server(priority.handle, str(priority_path), limit=MAX_REQUEST)
    try:
        os.chmod(socket_path, 0o600)
        os.chmod(priority_path, 0o600)
        async with server, control_server:
            while controller._session["state"] not in {"closing", "closed"}:
                await asyncio.sleep(0.1)
                try:
                    controller._live()
                except Refused:
                    break
    finally:
        server.close()
        control_server.close()
        await server.wait_closed()
        await control_server.wait_closed()
        await controller.retire()
        socket_path.unlink(missing_ok=True)
        priority_path.unlink(missing_ok=True)


async def main():
    # Keep Playwright's lifetime outside the readiness timeout and RPC lifetime.
    admission = process_evidence()
    os.umask(0o077)
    DIRECTORY.mkdir(mode=0o700, exist_ok=False)
    incarnation = secrets.token_hex(32)
    (DIRECTORY / "incarnation").write_text(incarnation)
    source = DIRECTORY / "bootstrap.json"
    async with asyncio.timeout(30):
        while not source.exists():
            await asyncio.sleep(0.05)
        with source.open("rb") as stream:
            raw = stream.read(MAX_REQUEST + 1)
        source.unlink()
        if len(raw) > MAX_REQUEST:
            raise Refused("invalid_bootstrap")
        settings = bootstrap(decode(raw))
        if not hmac.compare_digest(settings["incarnation"], incarnation):
            raise Refused("worker_incarnation_mismatch")
        remaining = (instant(settings["request"]["expires_at"]) - datetime.now(UTC)).total_seconds()
        if not 0 < remaining <= 3600:
            raise Refused("invalid_session_lifetime")
    from playwright.async_api import async_playwright
    async with async_playwright() as playwright:
        async with asyncio.timeout(20):
            browser = await playwright.chromium.launch(headless=True, chromium_sandbox=True, channel="chromium")
            admission.update(await chromium_evidence(browser))
            admission["worker_incarnation"] = incarnation
            (DIRECTORY / "admission.json").write_bytes(encode(admission))
            context = await browser.new_context(viewport={"width": 800, "height": 600},
                                                accept_downloads=False, service_workers="block")
            page = await context.new_page()
            from workspace import WorkspaceTransport
            controller = BrowserController(page, settings["request"], workspace_enabled=True)
            controller.workspace = WorkspaceTransport(controller, DIRECTORY / "workspace-transfer")
            await page.set_content(settings["document_html"], timeout=5000)
        signal.alarm(max(1, math.ceil(controller.deadline - controller.monotonic())) + 10)
        try:
            await serve(controller, settings["secret"])
        finally:
            async with asyncio.timeout(5):
                await browser.close()


if __name__ == "__main__":
    signal.alarm(45)
    try:
        asyncio.run(main())
    except Exception:  # noqa: BLE001 — boundary keeps private browser details out of IPC/logs
        # Process failure is physical custody for the host to reconcile. Never
        # echo bootstrap HTML, credentials or browser diagnostics to Docker logs.
        raise SystemExit(1) from None
