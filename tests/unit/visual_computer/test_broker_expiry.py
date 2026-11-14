"""Broker lifetime boundaries with real owned Tasks and explicitly scripted IO.

No subprocess, TLS, mount or physical retirement ACK is claimed here. The
controlled clock never replaces the real event loop's scheduling clock.
"""
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission, await_owned_task, create_owned_task,
)
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, execution_evidence_carriers,
)
from kestrel_sovereign.visual_computer import worker_broker as broker
from tests.unit.visual_computer.test_two_domain_worker_profile import settings


class Clock:
    def __init__(self):
        self.wall = datetime(2030, 1, 1, tzinfo=UTC)
        self.monotonic = 100.0
        self.timers = []
        self.readers = {}
        self.signals = {}

    def time(self):
        return self.monotonic

    def call_at(self, when, callback, *args):
        timer = SimpleNamespace(when=when, callback=callback, args=args, cancelled=False)
        timer.cancel = lambda: setattr(timer, "cancelled", True)
        self.timers.append(timer)
        return timer

    def call_later(self, seconds, callback, *args):
        return self.call_at(self.time() + seconds, callback, *args)

    def add_reader(self, fd, callback):
        self.readers[fd] = callback

    def remove_reader(self, fd):
        self.readers.pop(fd, None)

    def add_signal_handler(self, signum, callback):
        self.signals[signum] = callback

    def remove_signal_handler(self, signum):
        return self.signals.pop(signum, None) is not None


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()

    class WallClock(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is UTC
            return clock.wall

    class Asyncio:
        def __getattr__(self, name):
            return getattr(asyncio, name)

        def get_running_loop(self):
            return clock

    monkeypatch.setattr(broker, "datetime", WallClock)
    monkeypatch.setattr(broker, "asyncio", Asyncio())
    return clock


def configuration(clock, role=None):
    value = settings()
    value["expires_at"] = (clock.wall + timedelta(seconds=10)).isoformat()
    if role is not None:
        value["roles"] = {role.value: value["roles"][role.value]}
    return value


class Listener:
    """Scripted listener: no descriptor or kernel accept exists."""
    def __init__(self, failure=None):
        self.accepts = 0
        self.closed = 0
        self.failure = failure

    def fileno(self):
        return 17

    def setblocking(self, value):
        assert value is False

    def bind(self, address):
        self.address = address

    def listen(self, capacity):
        self.capacity = capacity

    def accept(self):
        self.accepts += 1
        raise self.failure or AssertionError("expired admission reached native accept")

    def close(self):
        self.closed += 1


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["wall-forward", "monotonic-after-wall-backward"])
async def test_expired_accept_revokes_without_poisoning_original_retirement(clock, step):
    lane = broker.WorkerLane(configuration(clock, broker.LaneRole.PRIORITY),
                             broker.LaneRole.PRIORITY, object())
    lane.listener = Listener()
    if step == "wall-forward":
        clock.wall += timedelta(seconds=11)
    else:
        clock.wall -= timedelta(hours=1)
        clock.monotonic += 11
    lane._accept_one()
    await lane._close_originals()
    assert not lane.active and lane.changed.is_set()
    assert lane.listener.accepts == 0 and lane.listener.closed == 1
    assert lane.terminal is None and all(slot is None for slot in lane.slots)


@pytest.mark.asyncio
async def test_lane_check_cannot_extend_lifetime_after_backward_wall_step(clock):
    lane = broker.WorkerLane(configuration(clock, broker.LaneRole.PRIORITY),
                             broker.LaneRole.PRIORITY, object())
    clock.wall -= timedelta(hours=1)
    clock.monotonic += 11
    with pytest.raises(RuntimeError, match="original_broker_lifetime_retired"):
        lane.check()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["check-control", "accept-error"])
async def test_unexpired_accept_retains_actual_control_and_native_errors(clock, kind):
    lane = broker.WorkerLane(configuration(clock, broker.LaneRole.PRIORITY),
                             broker.LaneRole.PRIORITY, object())
    original = (ExecutionAuthorityError("scripted original admission control")
                if kind == "check-control" else OSError("scripted original accept failure"))
    lane.listener = Listener(original)
    if kind == "check-control":
        def check():
            raise original
        lane.check = check
    lane._accept_one()
    with pytest.raises(BaseException) as caught:
        await lane._close_originals()
    assert original in execution_evidence_carriers(caught.value)
    assert lane.listener.closed == 1 and not lane.active


@pytest.mark.asyncio
async def test_lane_timer_keeps_original_monotonic_deadline_when_serve_starts_late(
        clock, monkeypatch):
    lane = broker.WorkerLane(configuration(clock, broker.LaneRole.PRIORITY),
                             broker.LaneRole.PRIORITY, object())
    listener, ready, ipc_closed = Listener(), asyncio.Event(), []
    ipc = SimpleNamespace(close=lambda: ipc_closed.append(True))

    class ReadyPath:
        def __init__(self, *args):
            pass

        def __truediv__(self, name):
            return self

        def write_text(self, value):
            assert value == "ready"
            ready.set()

    monkeypatch.setattr(broker, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(broker, "socket", SimpleNamespace(
        SO_PEERCRED=1, AF_INET=2, SOCK_STREAM=1, socket=lambda *args: listener))
    monkeypatch.setattr(broker, "require_ipc", lambda *args: ipc)
    monkeypatch.setattr(broker, "Path", ReadyPath)
    clock.wall -= timedelta(hours=1)
    clock.monotonic += 2
    task = create_owned_task(lane.serve(), name="original-scripted-expiry-lane")
    try:
        await asyncio.wait_for(ready.wait(), 2)
        assert len(clock.timers) == 1
        assert clock.timers[0].when == 110.0, "serve must use the original lifetime cap"
        clock.monotonic = 110.0
        clock.timers[0].callback(*clock.timers[0].args)
        outcome = await await_owned_task(task)
        assert outcome.error is None and outcome.cancellation is None
        assert listener.closed == 1 and ipc_closed == [True]
        assert clock.timers[0].cancelled and not clock.readers
    finally:
        lane.revoke()
        await await_owned_task(task)


class Child:
    """Scripted child handle; only wait Tasks and their joins are real."""
    def __init__(self, role):
        self.role, self.returncode = role, None
        self.exited = asyncio.Event()
        self.waits, self.terminations = 0, 0
        self.stdin = SimpleNamespace(write=lambda raw: None, close=lambda: None,
                                     drain=self.drain)

    async def drain(self):
        return None

    async def wait(self):
        await self.exited.wait()
        self.waits += 1
        return self.returncode

    def finish(self, code):
        self.returncode = code
        self.exited.set()

    def terminate(self):
        self.terminations += 1
        self.finish(0)


def script_children(clock, monkeypatch, on_ready):
    children, tasks = [], []
    real_create = broker.create_owned_task

    async def launch(*argv, **kwargs):
        child = Child(argv[-1])
        child.original_argv = argv
        children.append(child)
        return child

    async def ready(originals, retirement):
        assert originals == children and len(children) == 2
        on_ready(children)

    def create(*args, **kwargs):
        task = real_create(*args, **kwargs)
        tasks.append(task)
        return task

    monkeypatch.setattr(broker.asyncio, "create_subprocess_exec", launch, raising=False)
    monkeypatch.setattr(broker, "_wait_roles_ready", ready)
    monkeypatch.setattr(broker, "create_owned_task", create)
    return children, tasks


def program():
    return Path(broker.__file__).with_name("_worker_programs") / "broker_worker.py"


@pytest.mark.asyncio
async def test_supervisor_launches_each_lane_through_same_guarded_launcher(clock, monkeypatch):
    value=configuration(clock)
    def ready(children):
        clock.wall+=timedelta(seconds=11)
        clock.monotonic+=11
        children[0].finish(0)
    children,tasks=script_children(clock,monkeypatch,ready)
    await broker.supervise(value,program())
    for child in children:
        assert child.original_argv==(broker.sys.executable,"-B",
            str(Path(broker.__file__).with_name("worker_entry.py")),"broker","--lane",child.role)
    assert all(task.done()for task in tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["wall-forward", "monotonic-after-wall-backward"])
async def test_supervisor_accepts_zero_exit_only_at_actual_lifetime_boundary(
        clock, monkeypatch, step):
    value = configuration(clock)

    def ready(children):
        if step == "wall-forward":
            clock.wall += timedelta(seconds=11)
            clock.monotonic += 1
        else:
            clock.wall -= timedelta(hours=1)
            clock.monotonic += 11
        children[0].finish(0)

    children, tasks = script_children(clock, monkeypatch, ready)
    await broker.supervise(value, program())
    assert all(child.waits >= 1 and child.returncode == 0 for child in children)
    assert children[1].terminations == 1
    assert all(task.done() for task in tasks) and not clock.signals


@pytest.mark.asyncio
@pytest.mark.parametrize(("code", "expired"), [(0, False), (7, False), (7, True)])
async def test_supervisor_keeps_truly_early_or_nonzero_child_exit_fatal(
        clock, monkeypatch, code, expired):
    value = configuration(clock)

    def ready(children):
        if expired:
            clock.wall += timedelta(seconds=11)
            clock.monotonic += 11
        children[0].finish(code)

    children, tasks = script_children(clock, monkeypatch, ready)
    with pytest.raises(BaseException) as caught:
        await broker.supervise(value, program())
    assert any(isinstance(error, RuntimeError) and
               str(error) == "original_broker_child_ended"
               for error in execution_evidence_carriers(caught.value))
    assert all(child.waits >= 1 and child.returncode is not None for child in children)
    assert children[1].terminations == 1
    assert all(task.done() for task in tasks) and not clock.signals


@pytest.mark.asyncio
async def test_supervisor_original_timer_bounds_backward_wall_window(clock, monkeypatch):
    value, ready, shutdown = configuration(clock), asyncio.Event(), asyncio.Event()
    children, tasks = script_children(clock, monkeypatch, lambda _: ready.set())
    admission = OwnedTaskAdmission()
    owner = create_owned_task(broker.supervise(value, program(), _retirement=shutdown),
        admission=admission, name="original-scripted-expiry-supervisor")
    try:
        await asyncio.wait_for(ready.wait(), 2)
        await asyncio.sleep(0)
        assert len(clock.timers) == 1, "supervisor needs its own admitted lifetime cap"
        assert clock.timers[0].when == 110.0
        clock.wall -= timedelta(hours=1)
        clock.monotonic = 110.0
        clock.timers[0].callback(*clock.timers[0].args)
        outcome = await await_owned_task(owner)
        assert outcome.error is None and outcome.cancellation is None
        assert all(child.waits >= 1 and child.returncode == 0 for child in children)
        assert all(task.done() for task in tasks)
        assert clock.timers[0].cancelled and not clock.signals
    finally:
        shutdown.set()
        await await_owned_task(owner)


@pytest.mark.asyncio
async def test_observed_expiry_never_reopens_after_wall_clock_reversal(clock):
    lane = broker.WorkerLane(configuration(clock, broker.LaneRole.PRIORITY),
                             broker.LaneRole.PRIORITY, object())
    clock.wall += timedelta(seconds=11)
    with pytest.raises(RuntimeError, match="original_broker_lifetime_retired"):
        lane.check()
    clock.wall -= timedelta(hours=1)
    clock.monotonic += 1
    with pytest.raises(RuntimeError, match="original_broker_lifetime_retired"):
        lane.check()


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["wall-forward", "monotonic-cap"])
async def test_expiry_reached_at_original_check_entry_is_normal_retirement(clock, step):
    lane = broker.WorkerLane(configuration(clock, broker.LaneRole.PRIORITY),
                             broker.LaneRole.PRIORITY, object())
    lane.listener = Listener()
    original_check = lane.check
    def check():
        # Exact transition after the old precheck, before its original check.
        if step == "wall-forward":
            clock.wall += timedelta(seconds=11)
        else:
            clock.monotonic += 11
        original_check()
    lane.check = check
    lane._accept_one()
    await lane._close_originals()
    assert lane.listener.accepts == 0 and lane.listener.closed == 1
    assert lane.terminal is None and not lane.active
