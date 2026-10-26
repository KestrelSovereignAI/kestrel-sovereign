"""Authority revocation at real child-process stdin, separate from Docker proof."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys

import pytest
from kestrel_sdk.computer_use import ActionRequest, Click, ComputerSession
from kestrel_sdk.resources import ResourceOutcomeUnknown, ResourceScope
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
    program = f"#!{sys.executable}\nimport sys,time\nfrom pathlib import Path\n"
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
