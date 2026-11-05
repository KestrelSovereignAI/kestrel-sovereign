"""Offline browser domain: the existing controller/parser, Unix IPC only.

Installed as a normal Core resource. No TCP listener, network relay, files
transport or Human transport exists in this profile.
"""
import sys
from kestrel_sovereign.visual_computer.worker_entry import ProcessLifetime

_process_lifetime = None
if __name__ == "__main__" and sys.argv[1:] != ["--initialize"]:
    _process_lifetime = ProcessLifetime()
    if _process_lifetime.requested:
        raise SystemExit(0)  # No resource has been admitted.

import asyncio
from contextlib import asynccontextmanager
import hmac
import math
import os
from pathlib import Path
import secrets
import signal

from controller import BrowserController, Refused, instant
from sandbox_checks import chromium_evidence, process_evidence
from wire import MAX_REQUEST, decode, encode
from worker import Endpoint, bootstrap
from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission, await_owned_operation, await_owned_task, create_owned_task, raise_owned_outcome,
)
from kestrel_sovereign.execution_custody import execution_terminal_error
from kestrel_sovereign.visual_computer.worker_broker import require_peer
from kestrel_sovereign.visual_computer.worker_profile import (
    BROWSER_CLOSE_SECONDS, BROWSER_RETIRE_SECONDS, DRIVER, IPC, PRIVATE, TOKEN,
)
from kestrel_sovereign.visual_computer.worker_stream import join_original_writer_close as _join_original_writer_close


def configuration(value):
    if type(value) is not dict or type(value.get("profile_sha256")) is not str or TOKEN.fullmatch(value["profile_sha256"]) is None:
        raise Refused("invalid_bootstrap")
    original = dict(value)
    original.pop("profile_sha256")
    bootstrap(original)
    if original["network_profile"] is not None:
        raise Refused("unsupported_bootstrap_profile")
    return value


def initialize(raw):
    if len(raw) > MAX_REQUEST:
        raise Refused("invalid_bootstrap")
    value = configuration(decode(raw))
    directory = Path(PRIVATE)
    if not hmac.compare_digest(value["incarnation"], (directory/"incarnation").read_text()):
        raise Refused("worker_incarnation_mismatch")
    (directory/"initialized").touch(mode=0o600, exist_ok=False)
    pending = directory/"bootstrap.pending"
    with open(pending, "xb", opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
        stream.write(raw)
    os.rename(pending, directory/"bootstrap.json")
    return encode({"schema_version": 1, "submitted": True})


async def serve(controller, secret):
    """One controller/journal with the original finite Endpoint dispatch.

    Retain every actual server callback Task before coroutine publication.
    Socket peers must be the broker UID/GID; HMAC remains independently
    required by the unchanged canonical Endpoint parser.
    """
    endpoints = (Endpoint(controller, secret), Endpoint(controller, secret, priority=True))
    servers, admissions = [], []
    terminal = None
    active = True
    def accept(endpoint, reader, writer):
        nonlocal terminal, active
        admission = OwnedTaskAdmission()
        admissions.append(admission)
        async def invoke():
            failure = None
            try:
                if active:
                    require_peer(writer.get_extra_info("socket"), 10001, 10003)
                    await endpoint.handle(reader, writer)
            except BaseException as error:
                failure = error
            finally:
                errors = []
                try:
                    writer.close()
                except BaseException as error:
                    errors.append(error)
                outcome = await await_owned_operation(_join_original_writer_close(writer),
                                                       retirement_resource=writer, retain_originals=True)
                errors.extend(x for x in (outcome.error, outcome.cancellation) if x is not None)
                if failure is not None or errors:
                    raise execution_terminal_error(failure, *errors, retain_originals=True)
        try:
            create_owned_task(invoke(), admission=admission, retirement_resource=writer,
                              retain_originals=True, name="original-browser-ipc")
        except BaseException as error:
            terminal = execution_terminal_error(terminal, error, retain_originals=True)
            active = False
    try:
        for endpoint, name in zip(endpoints, ("worker.sock", "control.sock"), strict=True):
            path = Path(IPC)/name
            async def open_original(endpoint=endpoint, path=path):
                server = await asyncio.start_unix_server(lambda r, w: accept(endpoint, r, w), str(path), limit=MAX_REQUEST)
                servers.append(server)  # Original Server before publication.
                return server
            raise_owned_outcome(await await_owned_operation(open_original(), retirement_resource=servers),
                                operation="original browser IPC listener", retain_originals=True)
            os.chmod(path, 0o660)
        while controller._session["state"] not in {"closing", "closed"}:
            await asyncio.sleep(.1)
            controller._live()
            # Do not retain an unbounded history. Retire each actual callback
            # only after its canonical original join, never just Task.done.
            for admission in tuple(admissions):
                if admission.task is not None and admission.task.done():
                    outcome = await await_owned_task(admission.task)
                    raise_owned_outcome(outcome, operation="original browser IPC", retain_originals=True)
                    admissions.remove(admission)
            if terminal is not None:
                raise terminal
    except Refused:
        pass  # Fixed expiry closes this same controller; it does not renew.
    except BaseException as error:
        terminal = execution_terminal_error(terminal, error, retain_originals=True)
    finally:
        active = False
        for server in servers:
            try:
                server.close()
            except BaseException as error:
                terminal = execution_terminal_error(terminal, error, retain_originals=True)
        for server in servers:
            try:
                await server.wait_closed()
            except BaseException as error:
                terminal = execution_terminal_error(terminal, error, retain_originals=True)
        # Listener close can synchronously publish a last accepted writer.
        # Drain the original callbacks retained through the serving join.
        while admissions:
            originals = tuple(admissions)
            for admission in originals:
                if admission.task is not None:
                    outcome = await await_owned_task(admission.task)
                    terminal = execution_terminal_error(terminal, outcome.error, outcome.cancellation, retain_originals=True)
                admissions.remove(admission)
        try:
            await controller.retire()
        except BaseException as error:
            terminal = execution_terminal_error(terminal, error, retain_originals=True)
        for name in ("worker.sock", "control.sock"):
            try:
                (Path(IPC)/name).unlink(missing_ok=True)
            except BaseException as error:
                terminal = execution_terminal_error(terminal, error, retain_originals=True)
        if terminal is not None:
            raise terminal


@asynccontextmanager
async def original_playwright():
    """Own the pinned canonical Playwright connection, init Task and Process.

    Playwright's public manager drops its pump Task. This finite pinned adapter
    runs that SAME Connection/PipeTransport program with an explicit original
    admission. It does not replace its parser, object factory or driver. No
    Chromium Process.wait receipt is invented; kernel/cgroup proof is separate.
    """
    from importlib.metadata import version
    from playwright._impl._connection import Connection
    from playwright._impl._object_factory import create_remote_object
    from playwright._impl._transport import PipeTransport
    from playwright.async_api._generated import Playwright as AsyncPlaywright
    if version("playwright") != "1.63.0":
        raise RuntimeError("pinned_playwright_ownership_program_required")
    loop = asyncio.get_running_loop()
    class OriginalTransport(PipeTransport):
        def __init__(self):
            super().__init__(loop)
            self.opened = loop.create_future()
        async def connect(self):
            try:
                # The original implementation retains its actual Process on
                # this transport before publishing its successful return.
                outcome = await await_owned_operation(super().connect(), retirement_resource=self,
                                                      retain_originals=True, name="original-playwright-connect")
                return raise_owned_outcome(outcome, operation="original Playwright pipe open", retain_originals=True)
            finally:
                self.opened.set_result(None)
    transport = OriginalTransport()
    connection = Connection(None, create_remote_object, transport, loop)
    pump = OwnedTaskAdmission()
    failure = None
    try:
        create_owned_task(connection.run(), admission=pump, retirement_resource=connection,
                          retain_originals=True, name="original-playwright-pump")
        async def ready():
            async with asyncio.timeout(20):
                done, _ = await asyncio.wait((transport.on_error_future, connection.playwright_future),
                                            return_when=asyncio.FIRST_COMPLETED)
                if not connection.playwright_future.done():
                    connection.playwright_future.cancel()
                return AsyncPlaywright(next(iter(done)).result())
        playwright = raise_owned_outcome(await await_owned_operation(ready(), retirement_resource=connection,
            retain_originals=True, cancel_on_caller_cancellation=True, name="original-playwright-ready"),
            operation="original Playwright startup", retain_originals=True)
        yield playwright
    except BaseException as error:
        failure = error
    finally:
        errors = []
        async def joined(operation, resource):
            outcome = await await_owned_operation(operation, retirement_resource=resource, retain_originals=True)
            errors.extend(x for x in (outcome.error, outcome.cancellation) if x is not None)
            return outcome
        if pump.task is None:
            errors.append(RuntimeError("original_playwright_pump_admission_unresolved"))
        else:
            # Caller withdrawal cannot stop an opening pipe before the
            # original native Process constructor has settled and published.
            await joined(asyncio.wait((pump.task, transport.opened), return_when=asyncio.FIRST_COMPLETED), connection)
            await joined(connection.stop_async(), connection)
            node = getattr(transport, "_proc", None)
            if node is not None and type(node) is asyncio.subprocess.Process:
                # Even a failed canonical stop must withdraw the original
                # pipe and join the pump/process rather than skip their waits.
                try:
                    node.stdin.close()
                except BaseException as error:
                    errors.append(error)
            elif transport.opened.done() and node is not None:
                errors.append(RuntimeError("original_playwright_process_identity_unknown"))
            outcome = await await_owned_task(pump.task)
            errors.extend(x for x in (outcome.error, outcome.cancellation) if x is not None)
            init = connection._init_task
            if init is not None:
                outcome = await await_owned_task(init)
                errors.extend(x for x in (outcome.error, outcome.cancellation) if x is not None)
            elif node is not None:
                errors.append(RuntimeError("original_playwright_init_admission_unresolved"))
            if node is not None and type(node) is asyncio.subprocess.Process:
                outcome = await joined(node.wait(), node)
                if outcome.error is None and outcome.result != 0:
                    errors.append(RuntimeError("original_playwright_driver_native_exit_unknown"))
        for future in (connection.playwright_future, transport.on_error_future):
            if future.done() and not future.cancelled():
                error = future.exception()
                if error is not None:
                    errors.append(error)
        if failure is not None or errors:
            raise execution_terminal_error(failure, *errors, retain_originals=True)


async def main(lifecycle):
    return await ProcessLifetime.run(lifecycle, _main(lifecycle))


async def _main(lifecycle):
    if lifecycle.requested:
        return
    admission = process_evidence()
    if os.getuid() != 10002 or os.getgid() != 10003:
        raise Refused("browser_role_identity_required")
    os.umask(0o077)
    loop, retirement, controller = asyncio.get_running_loop(), asyncio.Event(), None
    retirement_started = False
    def arm_retirement():
        nonlocal retirement_started
        if not retirement_started:
            retirement_started = True
            signal.alarm(BROWSER_RETIRE_SECONDS)
    def retire_original():
        retirement.set()
        arm_retirement()
        if controller is not None:
            controller._fence_retirement()
    lifecycle.bind(loop, retire_original)
    if lifecycle.requested:
        return
    directory = Path(PRIVATE)
    directory.mkdir(mode=0o700, exist_ok=False)
    incarnation = secrets.token_hex(32)
    (directory/"incarnation").write_text(incarnation)
    source = directory/"bootstrap.json"
    async with asyncio.timeout(30):
        while not source.exists() and not retirement.is_set():
            await asyncio.sleep(.05)
        if retirement.is_set():
            return
        with source.open("rb") as stream:
            raw = stream.read(MAX_REQUEST+1)
        source.unlink()
        settings = configuration(decode(raw))
        if not hmac.compare_digest(settings["incarnation"], incarnation):
            raise Refused("worker_incarnation_mismatch")
    shared = Path(IPC)
    for name, value in (("incarnation", incarnation), ("profile", settings["profile_sha256"])):
        with open(shared/name, "x", opener=lambda p, flags: os.open(p, flags, 0o640)) as stream:
            stream.write(value)
            os.fchmod(stream.fileno(), 0o640)
    async with original_playwright() as playwright:
        browser, failure = None, None
        try:
            async with asyncio.timeout(20):
                async def launch_original():
                    nonlocal browser
                    browser = await playwright.chromium.launch(headless=True, chromium_sandbox=True, channel="chromium")
                    return browser  # Capture before fallible publication.
                raise_owned_outcome(await await_owned_operation(launch_original(), retirement_resource=playwright,
                    retain_originals=True, name="original-chromium-open"),
                    operation="original Chromium startup", retain_originals=True)
                if retirement.is_set():
                    return
                admission.update(await chromium_evidence(browser))
                admission["worker_incarnation"] = incarnation
                admission["gid"] = os.getgid()
                context = await browser.new_context(viewport={"width": 800, "height": 600}, accept_downloads=False, service_workers="block")
                page = await context.new_page()
                controller = BrowserController(page, settings["request"], driver_id=DRIVER,
                                               workspace_enabled=False, network_enabled=False)
                if retirement.is_set():
                    controller._fence_retirement()
                async def deny_socket(connection):
                    await connection.close(code=1008, reason="unsupported_channel")
                await context.route_web_socket("**/*", deny_socket)
                await context.add_init_script("""for(const name of ['RTCPeerConnection','webkitRTCPeerConnection']){
                    Object.defineProperty(globalThis,name,{value:class{constructor(){throw new DOMException('unsupported_channel','SecurityError')}},writable:false,configurable:false});}""")
                async def deny_request(route):
                    await route.abort("blockedbyclient")
                await context.route("**/*", deny_request)
                await page.set_content(settings["document_html"], timeout=5000)
                (directory/"admission.json").write_bytes(encode(admission))
            if not retirement_started:
                signal.alarm(max(1, math.ceil(controller.deadline-controller.monotonic()))+BROWSER_RETIRE_SECONDS)
            await serve(controller, settings["secret"])
        except BaseException as error:
            failure = error
        finally:
            errors = []
            # Setup failure and natural expiry enter the same bounded original
            # cleanup as lifecycle withdrawal. This grants no action lifetime.
            arm_retirement()
            if browser is not None:
                async def close_original():
                    async with asyncio.timeout(BROWSER_CLOSE_SECONDS):
                        await browser.close()
                outcome = await await_owned_operation(close_original(), retirement_resource=browser, retain_originals=True)
                errors.extend(x for x in (outcome.error, outcome.cancellation) if x is not None)
            if failure is not None or errors:
                raise execution_terminal_error(failure, *errors, retain_originals=True)


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["--initialize"]:
            sys.stdout.buffer.write(initialize(sys.stdin.buffer.read(MAX_REQUEST+1)))
        elif not sys.argv[1:]:
            signal.alarm(45)
            asyncio.run(main(_process_lifetime))
        else:
            raise Refused("fixed_browser_role_required")
    except Exception:  # noqa: BLE001 — private bootstrap/diagnostics never enter logs
        raise SystemExit(1) from None
