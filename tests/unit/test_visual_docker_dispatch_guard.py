"""Authority revocation at real child-process stdin, separate from Docker proof."""
import asyncio
import ast
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
import inspect
import sys
import textwrap
import time

import pytest
from kestrel_sdk.computer_use import ActionRequest, Click, ComputerSession
from kestrel_sdk.resources import ResourceOutcomeUnknown, ResourceScope
from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ConfirmedLeaseDeadline, execution_cleanup_debts, execution_error_with_cause,
    execution_evidence_carriers, execution_native_source_error,
    execution_terminal_error, is_execution_control_error,
)
from kestrel_sovereign.visual_computer.docker import DockerBoundary, DockerComputerDriver, PROVIDER


class Revoked(Exception):
    pass


def current_check(active):
    def current():
        if not active[0]:
            raise Revoked("original_authority_revoked")
    return current


def executable(tmp_path, *, delayed=False):
    path = tmp_path / "owned-stdin-fixture"
    program = f"#!{sys.executable} -B\nimport sys,time\nfrom pathlib import Path\n"
    program += f"data=sys.stdin.buffer.read(); p=Path({str(tmp_path / 'accepted')!r})\n"
    program += "if data: p.write_bytes(data)\n"
    if delayed:
        program += f"gate=Path({str(tmp_path / 'reply')!r})\nwhile not gate.exists(): time.sleep(.005)\n"
    program += "sys.stdout.buffer.write(b'{}')\n"
    path.write_text(program); path.chmod(0o700)
    return path


@pytest.mark.asyncio
async def test_revocation_during_process_launch_prevents_actual_stdin_effect(tmp_path, monkeypatch):
    boundary = DockerBoundary("/tmp/fixture.sock")
    boundary.executable = str(executable(tmp_path))
    active, launched, proceed, processes = [True], asyncio.Event(), asyncio.Event(), []
    original = asyncio.create_subprocess_exec
    async def launch(*arguments, **options):
        process = await original(*arguments, **options)
        processes.append(process); launched.set()
        await proceed.wait()
        return process
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    pending = asyncio.create_task(boundary.invoke(["exec", "fixture"], data=b"SYNTHETIC INPUT", current=current_check(active)))
    await launched.wait()
    active[0] = False; proceed.set()
    with pytest.raises(Revoked):
        await pending
    assert not (tmp_path / "accepted").exists()
    assert processes[0].returncode is not None


@pytest.mark.asyncio
async def test_cancel_during_process_launch_joins_exact_child_without_sending_input(tmp_path,monkeypatch):
    boundary = DockerBoundary("/tmp/fixture.sock")
    boundary.executable = str(executable(tmp_path))
    launched,proceed,processes = asyncio.Event(),asyncio.Event(),[]
    original = asyncio.create_subprocess_exec
    async def launch(*arguments,**options):
        process = await original(*arguments,**options)
        processes.append(process);launched.set()
        await proceed.wait()
        return process
    monkeypatch.setattr(asyncio,"create_subprocess_exec",launch)
    pending = asyncio.create_task(boundary.invoke(["exec","fixture"],data=b"SYNTHETIC INPUT"))
    await launched.wait();pending.cancel();await asyncio.sleep(0)
    assert not pending.done()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert processes[0].returncode is not None and not (tmp_path/"accepted").exists()


@pytest.mark.asyncio
async def test_revocation_after_stdin_is_unknown_and_owned_process_is_joined(tmp_path):
    boundary = DockerBoundary("/tmp/fixture.sock")
    boundary.executable = str(executable(tmp_path, delayed=True))
    active = [True]
    pending = asyncio.create_task(boundary.invoke(["exec", "fixture"], data=b"SYNTHETIC INPUT", current=current_check(active)))
    async with asyncio.timeout(3):
        while not (tmp_path / "accepted").exists():
            await asyncio.sleep(.005)
    active[0] = False
    (tmp_path / "reply").touch()
    with pytest.raises(ResourceOutcomeUnknown, match="effect_outcome_unknown"):
        await pending
    assert (tmp_path / "accepted").read_bytes() == b"SYNTHETIC INPUT"


@pytest.mark.asyncio
async def test_original_authority_survives_unknown_after_actual_stdin_acceptance(tmp_path):
    boundary = DockerBoundary("/tmp/fixture.sock")
    boundary.executable = str(executable(tmp_path, delayed=True))
    lease = ConfirmedLeaseDeadline()
    assert lease.confirm(time.monotonic() + 30)
    original_errors = []

    def current():
        try:
            lease.require_work()
        except BaseException as error:
            original_errors.append(error)
            raise

    payload = b"original controlled stdin\n"
    pending = asyncio.create_task(boundary.invoke(
        ["exec", "fixture"], data=payload, current=current))
    try:
        async with asyncio.timeout(3):
            while not (tmp_path / "accepted").exists():
                await asyncio.sleep(.005)
        assert (tmp_path / "accepted").read_bytes() == payload
        lease.mark_lost()
        (tmp_path / "reply").touch()
        with pytest.raises(ResourceOutcomeUnknown, match="driver_effect_outcome_unknown") as caught:
            await pending
        assert len(original_errors) == 1
        assert is_execution_control_error(caught.value), "UNKNOWN erased original authority loss"
        assert any(error is original_errors[0]
                   for error in execution_evidence_carriers(caught.value))
        assert execution_cleanup_debts(caught.value) == ()
    finally:
        (tmp_path / "reply").touch()
        if not pending.done():
            pending.cancel()
        await await_owned_task(pending)


@pytest.mark.asyncio
async def test_native_reply_timeout_retains_original_timeout_without_authority_loss(tmp_path):
    boundary = DockerBoundary("/tmp/fixture.sock")
    boundary.executable = str(executable(tmp_path, delayed=True))
    payload = b"bounded ordinary stdin\n"
    with pytest.raises(ResourceOutcomeUnknown, match="docker_transport_outcome_unknown") as caught:
        await boundary.invoke(["exec", "fixture"], data=payload, timeout=.5)
    assert (tmp_path / "accepted").read_bytes() == payload
    assert isinstance(caught.value.__cause__, TimeoutError), "UNKNOWN erased original native timeout"
    assert not is_execution_control_error(caught.value)
    assert execution_cleanup_debts(caught.value) == ()


@pytest.mark.asyncio
async def test_unknown_preserves_native_source_without_latching_observer_cancellation(tmp_path):
    boundary = DockerBoundary("/tmp/fixture.sock")
    boundary.executable = str(executable(tmp_path, delayed=True))
    lease = ConfirmedLeaseDeadline()
    assert lease.confirm(time.monotonic() + 30)
    receipt = [None]
    check_gate, observing = asyncio.Event(), asyncio.Event()

    def current():
        if receipt[0] is not None:
            raise receipt[0]
        lease.require_work()

    async def original_check():
        await check_gate.wait()
        lease.require_work()

    async def observe(original):
        observing.set()
        return await await_owned_task(original)

    payload = b"finite observer controlled stdin\n"
    pending = asyncio.create_task(boundary.invoke(
        ["exec", "fixture"], data=payload, current=current))
    original = observer = None
    try:
        async with asyncio.timeout(3):
            while not (tmp_path / "accepted").exists():
                await asyncio.sleep(.005)
        assert (tmp_path / "accepted").read_bytes() == payload
        original = asyncio.create_task(original_check())
        observer = asyncio.create_task(observe(original))
        await observing.wait()
        observer.cancel()
        await asyncio.sleep(0)
        assert not observer.done()
        lease.mark_lost()
        check_gate.set()
        outcome = await observer
        assert is_execution_control_error(outcome.error)
        assert isinstance(outcome.cancellation, asyncio.CancelledError)
        receipt[0] = execution_terminal_error(
            outcome.error, outcome.cancellation,
            retain_originals=True, isolate_control_originals=True)
        assert execution_native_source_error(receipt[0]) is outcome.error
        (tmp_path / "reply").touch()
        with pytest.raises(ResourceOutcomeUnknown, match="driver_effect_outcome_unknown") as caught:
            await pending
        carriers = execution_evidence_carriers(caught.value)
        assert any(error is receipt[0] for error in carriers)
        assert any(error is outcome.error for error in carriers)
        assert any(error is outcome.cancellation for error in carriers)
        projected = execution_native_source_error(caught.value)
        assert projected is outcome.error, "UNKNOWN changed the original finite native source"
        assert all(error is not outcome.cancellation
                   for error in execution_evidence_carriers(projected))
        assert execution_cleanup_debts(caught.value) == ()
    finally:
        check_gate.set()
        (tmp_path / "reply").touch()
        for owned in (original, observer, pending):
            if owned is not None:
                if not owned.done():
                    owned.cancel()
                await await_owned_task(owned)


def test_private_reply_revocation_preserves_original_carrier_static_wiring():
    # Static wiring only: the live driver requires its real Docker identity and
    # admission. Do not replace that boundary with a fabricated RPC/connection.
    function = DockerComputerDriver._rpc
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    handlers = [node for node in ast.walk(tree)
                if isinstance(node, ast.ExceptHandler)
                and isinstance(node.type, ast.Name) and node.type.id == "Exception"]
    assert len(handlers) == 1
    handler = handlers[0]
    raises = [node for node in ast.walk(handler) if isinstance(node, ast.Raise)]
    assert len(raises) == 1
    raised = raises[0]
    assert raised.cause is None
    call = raised.exc
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    assert call.func.id == "execution_error_with_cause"
    assert function.__globals__[call.func.id] is execution_error_with_cause
    assert len(call.args) == 2 and not call.keywords
    wrapper, original = call.args
    assert isinstance(wrapper, ast.Call) and isinstance(wrapper.func, ast.Name)
    assert function.__globals__[wrapper.func.id] is ResourceOutcomeUnknown
    assert len(wrapper.args) == 1 and isinstance(wrapper.args[0], ast.Constant)
    assert wrapper.args[0].value == "driver_effect_outcome_unknown"
    assert isinstance(original, ast.Name) and original.id == handler.name


@pytest.mark.asyncio
async def test_revocation_during_inspection_does_not_dispatch_action_and_joins_borrower(monkeypatch):
    active, borrowers, inspected, proceed = [True], [], asyncio.Event(), asyncio.Event()
    now = datetime.now(UTC)
    session = ComputerSession(session_id="session", scope=ResourceScope(tenant_id="fixture", principal_id="owner"),
        lease_id="lease", driver_id=PROVIDER, surface="browser", capabilities=frozenset({"screenshot", "click"}),
        state="ready", controller="agent", control_epoch=0, expires_at=now+timedelta(seconds=30))
    action = ActionRequest(action_id="action", session_id="session", observation_id="frame", control_epoch=0,
                           not_after=now+timedelta(seconds=5), action=Click(x=1, y=1))
    origin = asyncio.current_task()
    @asynccontextmanager
    async def guard(value, action_id):
        assert value == session and action_id == "action" and asyncio.current_task() is not origin
        borrowers.append(asyncio.current_task())
        try:
            yield current_check(active)
        finally:
            borrowers.remove(asyncio.current_task())
    driver = DockerComputerDriver(docker_socket="/tmp/fixture.sock", egress_guard=guard)
    driver._connections["session"] = object()
    async def bound(connection, **options):
        inspected.set(); await proceed.wait()
        return {"container_id": "c"*64}
    dispatched = []
    async def rpc(*args, **options):
        dispatched.append(args)
        raise AssertionError("revoked action reached credential dispatch")
    monkeypatch.setattr(driver, "_bound", bound)
    monkeypatch.setattr(driver.boundary, "rpc", rpc)
    pending = asyncio.create_task(driver.act(session, action))
    await inspected.wait()
    active[0] = False; proceed.set()
    with pytest.raises(Revoked):
        await pending
    assert dispatched == [] and borrowers == [] and driver._borrowers == {}
