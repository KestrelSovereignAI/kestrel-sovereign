"""Pre-admission deadlines in real packaged programs; no native-worker proof."""
import asyncio
import os
from types import SimpleNamespace

import pytest

from tests.unit.visual_computer.test_worker_entry import (
    canonical_private_worker_programs as canonical_private_worker_programs,
    process_signals as process_signals,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["browser", "broker"])
async def test_bootstrap_deadline_retires_before_configuration_or_native_admission(
    monkeypatch, tmp_path, canonical_private_worker_programs, role,
):
    program = canonical_private_worker_programs[
        "two_domain_worker" if role == "browser" else "broker_worker"
    ]
    events, budgets = [], []
    private = tmp_path / "private"
    monkeypatch.setattr(program, "PRIVATE", str(private) if role == "browser" else private)
    monkeypatch.setattr(program, "os", SimpleNamespace(
        getuid=lambda: 10002 if role == "browser" else 10001,
        getgid=lambda: 10003, umask=os.umask,
    ))
    monkeypatch.setattr(program.sys, "argv", [str(program.__file__)])
    if role == "browser":
        monkeypatch.setattr(program, "process_evidence", lambda: {})

    def forbidden(*_args, **_kwargs):
        events.append("configuration-or-native-admission")
        pytest.fail("an unbootstrapped worker entered configuration or native admission")

    if role == "browser":
        monkeypatch.setattr(program, "configuration", forbidden)
        monkeypatch.setattr(program, "original_playwright", forbidden)
    else:
        monkeypatch.setattr(program, "validate_configuration", forbidden)
        monkeypatch.setattr(program, "supervise", forbidden)

    timeout = asyncio.timeout

    def bounded(seconds):
        budgets.append(seconds)
        return timeout(.01 if seconds == 30 else seconds)

    monkeypatch.setattr(asyncio, "timeout", bounded)
    lifetime = SimpleNamespace(
        requested=False,
        bind=lambda _loop, _callback: events.append("original-lifetime-bound"),
        detach=lambda: events.append("original-lifetime-detached"),
    )
    previous = os.umask(0o077)
    try:
        result = await program.main(lifetime)
    finally:
        os.umask(previous)
    assert result is None, "a pre-admission bootstrap deadline did not retire cleanly"
    assert budgets == [30]
    assert events == ["original-lifetime-bound", "original-lifetime-detached"]
    assert (private / "incarnation").is_file()
    assert not (private / "bootstrap.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("withdrawal", ["caller", "process-latch"])
async def test_bootstrap_wait_preserves_original_cancellation_or_process_retirement(
    tmp_path, process_signals, withdrawal,
):
    from kestrel_sovereign._async_ownership import (
        OwnedTaskAdmission, await_owned_task, create_owned_task,
    )
    from kestrel_sovereign.visual_computer.worker_entry import ProcessLifetime

    lifetime = ProcessLifetime()
    entered = asyncio.Event()
    path = tmp_path / "never-admitted-bootstrap.json"

    async def original_wait():
        entered.set()
        return await ProcessLifetime.wait_for_bootstrap(lifetime, path, asyncio.Event())

    admission = OwnedTaskAdmission()
    task = create_owned_task(original_wait(), admission=admission,
                             retain_originals=True, name="original-bootstrap-wait")
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if withdrawal == "caller":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled(), "bootstrap waiting swallowed its caller's cancellation"
        else:
            # This is the real process latch; no async callback has been bound.
            lifetime._request(None, None)
            assert await asyncio.wait_for(task, 2) is False
        assert not path.exists()
    finally:
        if not task.done():
            task.cancel()
        await await_owned_task(task)
