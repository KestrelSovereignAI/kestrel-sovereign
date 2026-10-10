"""``CodexAppServerClient.aclose`` forgets its process only once it has exited (#3559).

These run a real child process as the app-server. A close that could not
confirm the process exited raises and keeps the process, so a later close
retries it; the adapter above keeps its client the same way.
"""

import asyncio
import sys

import pytest

from kestrel_sovereign.llm import codex_app_server
from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient

#: Exits once its stdin is closed, as the app-server does.
_EXITS_ON_EOF = "import sys; sys.stdin.read()"
#: Keeps running after its stdin is closed, until it is killed.
_IGNORES_EOF = "import sys, time; sys.stdin.read(); time.sleep(600)"
#: Closes its stdout at once, as a dying app-server does, but keeps running.
_CLOSES_STDOUT_AND_RUNS = "import os, sys, time; os.close(1); sys.stdin.read(); time.sleep(600)"


@pytest.fixture
async def spawn():
    procs: list[asyncio.subprocess.Process] = []

    async def _spawn(script: str, *, stdout=asyncio.subprocess.DEVNULL):
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            stdin=asyncio.subprocess.PIPE,
            stdout=stdout,
            stderr=asyncio.subprocess.DEVNULL,
        )
        procs.append(proc)
        client = CodexAppServerClient(binary="codex-under-test")
        client._proc = proc
        client._initialized = True
        return client

    yield _spawn
    for proc in procs:
        if proc.returncode is None:
            # Past any kill a test patched onto the instance.
            asyncio.subprocess.Process.kill(proc)
            await proc.wait()


@pytest.fixture
def short_grace(monkeypatch):
    monkeypatch.setattr(codex_app_server, "CODEX_APP_SERVER_EXIT_GRACE_S", 0.05)


@pytest.mark.asyncio
async def test_a_process_that_exits_when_asked_is_forgotten(spawn):
    client = await spawn(_EXITS_ON_EOF)
    proc = client._proc

    await client.aclose()

    assert proc.returncode == 0
    assert client._proc is None
    assert client._initialized is False


@pytest.mark.asyncio
async def test_a_process_that_ignores_the_request_is_killed_and_forgotten(
    spawn, short_grace
):
    client = await spawn(_IGNORES_EOF)
    proc = client._proc

    await client.aclose()

    assert proc.returncode is not None and proc.returncode != 0
    assert client._proc is None


@pytest.mark.asyncio
async def test_a_failed_kill_keeps_the_process_for_a_retry(
    spawn, short_grace, monkeypatch
):
    client = await spawn(_IGNORES_EOF)
    proc = client._proc
    real_kill = proc.kill
    kills = []

    def refuse_kill():
        kills.append("refused")
        raise PermissionError("injected kill failure")

    monkeypatch.setattr(proc, "kill", refuse_kill)
    with pytest.raises(PermissionError, match="injected kill failure"):
        await client.aclose()

    assert kills == ["refused"]
    assert proc.returncode is None, "the process is still running"
    assert client._proc is proc, "kept for a retry"

    monkeypatch.setattr(proc, "kill", real_kill)
    await client.aclose()

    assert proc.returncode is not None
    assert client._proc is None


@pytest.mark.asyncio
async def test_a_killed_process_still_running_keeps_the_process(
    spawn, short_grace, monkeypatch
):
    client = await spawn(_IGNORES_EOF)
    proc = client._proc
    real_kill = proc.kill
    monkeypatch.setattr(codex_app_server, "CODEX_APP_SERVER_KILL_WAIT_S", 0.05)
    monkeypatch.setattr(proc, "kill", lambda: None)

    with pytest.raises(TimeoutError):
        await client.aclose()

    assert proc.returncode is None
    assert client._proc is proc

    monkeypatch.setattr(proc, "kill", real_kill)
    await client.aclose()

    assert proc.returncode is not None
    assert client._proc is None


@pytest.mark.asyncio
async def test_a_cancelled_close_keeps_the_process(spawn, monkeypatch):
    client = await spawn(_IGNORES_EOF)
    proc = client._proc

    closing = asyncio.create_task(client.aclose())
    await asyncio.sleep(0.05)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert proc.returncode is None
    assert client._proc is proc

    monkeypatch.setattr(codex_app_server, "CODEX_APP_SERVER_EXIT_GRACE_S", 0.05)
    await client.aclose()

    assert proc.returncode is not None
    assert client._proc is None


@pytest.mark.asyncio
async def test_a_process_the_read_loop_forgot_is_kept_when_the_close_fails(
    spawn, short_grace, monkeypatch
):
    """Its stdout ending makes the read loop forget it, alive or not."""
    client = await spawn(_IGNORES_EOF)
    proc = client._proc
    real_kill = proc.kill

    def stdout_ended_then_kill_failed():
        client._forget_proc()
        raise PermissionError("injected kill failure")

    monkeypatch.setattr(proc, "kill", stdout_ended_then_kill_failed)
    with pytest.raises(PermissionError):
        await client.aclose()

    assert proc.returncode is None
    assert client._unexited_procs == [proc], "kept for a retry"

    monkeypatch.setattr(proc, "kill", real_kill)
    await client.aclose()

    assert proc.returncode is not None
    assert client._unexited_procs == []


@pytest.mark.asyncio
async def test_a_running_process_the_read_loop_forgot_is_still_stopped(
    spawn, short_grace
):
    """A process whose stdout ended but which kept running is not closed yet."""
    client = await spawn(_IGNORES_EOF)
    proc = client._proc
    client._forget_proc()
    assert client._proc is None

    await client.aclose()

    assert proc.returncode is not None
    assert client._unexited_procs == []


@pytest.mark.asyncio
async def test_a_process_a_respawn_replaced_is_still_stopped(spawn, short_grace):
    client = await spawn(_IGNORES_EOF)
    replaced = client._proc
    replacement = (await spawn(_IGNORES_EOF))._proc
    client._forget_proc()
    client._proc = replacement

    await client.aclose()

    assert replaced.returncode is not None
    assert replacement.returncode is not None
    assert client._proc is None
    assert client._unexited_procs == []


@pytest.mark.asyncio
async def test_the_read_loop_keeps_a_process_whose_stdout_ended_while_it_runs(
    spawn, short_grace
):
    client = await spawn(_CLOSES_STDOUT_AND_RUNS, stdout=asyncio.subprocess.PIPE)
    proc = client._proc

    await client._read_loop()

    assert client._proc is None, "a request may spawn a fresh app-server"
    assert client._initialized is False
    assert proc.returncode is None
    assert client._unexited_procs == [proc]

    await client.aclose()

    assert proc.returncode is not None
    assert client._unexited_procs == []


@pytest.mark.asyncio
async def test_a_respawn_keeps_the_process_it_replaces(
    spawn, short_grace, monkeypatch, tmp_path
):
    """A failed handshake leaves its process running; a respawn must not lose it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    client = await spawn(_IGNORES_EOF)
    replaced = client._proc
    replacement = (await spawn(_IGNORES_EOF))._proc

    async def create_subprocess_exec(*_args, **_kwargs):
        return replacement

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    await client._spawn()
    for task in (client._reader_task, client._stderr_task):
        task.cancel()

    assert client._proc is replacement
    assert client._unexited_procs == [replaced]

    await client.aclose()

    assert replaced.returncode is not None
    assert replacement.returncode is not None
    assert client._unexited_procs == []


@pytest.mark.asyncio
async def test_the_current_process_forgotten_mid_close_is_still_stopped(
    spawn, short_grace, monkeypatch
):
    """The read loop can forget it while an earlier process is being stopped."""
    client = await spawn(_IGNORES_EOF)
    earlier = client._proc
    client._forget_proc()
    current = (await spawn(_IGNORES_EOF))._proc
    client._proc = current
    real_stop = CodexAppServerClient._stop_process

    async def stop_while_the_read_loop_forgets(proc):
        if proc is earlier:
            client._forget_proc()
        await real_stop(proc)

    monkeypatch.setattr(
        CodexAppServerClient,
        "_stop_process",
        staticmethod(stop_while_the_read_loop_forgets),
    )

    await client.aclose()

    assert earlier.returncode is not None
    assert current.returncode is not None
    assert client._proc is None
    assert client._unexited_procs == []


@pytest.mark.asyncio
async def test_a_close_waits_for_a_spawn_in_progress_and_stops_it(
    spawn, short_grace
):
    client = await spawn(_IGNORES_EOF)
    first = client._proc
    await client._start_lock.acquire()
    closing = asyncio.create_task(client.aclose())
    await asyncio.sleep(0.05)
    assert not closing.done(), "the close waits for the spawn"

    # The spawn in progress replaces the process, then releases the lock.
    client._forget_proc()
    spawned = (await spawn(_IGNORES_EOF))._proc
    client._proc = spawned
    client._start_lock.release()
    await asyncio.wait_for(closing, timeout=10.0)

    assert first.returncode is not None
    assert spawned.returncode is not None
    assert client._proc is None
    assert client._unexited_procs == []


@pytest.mark.asyncio
async def test_a_process_that_cannot_be_stopped_does_not_keep_the_rest_running(
    spawn, short_grace, monkeypatch
):
    client = await spawn(_IGNORES_EOF)
    stuck = client._proc
    client._forget_proc()
    current = (await spawn(_IGNORES_EOF))._proc
    client._proc = current
    real_kill = stuck.kill

    def refuse_kill():
        raise PermissionError("injected kill failure")

    monkeypatch.setattr(stuck, "kill", refuse_kill)
    with pytest.raises(PermissionError):
        await client.aclose()

    assert current.returncode is not None, "the live process was still stopped"
    assert stuck.returncode is None
    assert client._unexited_procs == [stuck]

    monkeypatch.setattr(stuck, "kill", real_kill)
    await client.aclose()

    assert stuck.returncode is not None
    assert client._proc is None
    assert client._unexited_procs == []


@pytest.mark.asyncio
async def test_the_read_loop_finishes_before_the_close_returns(spawn, short_grace):
    """Its cleanup must not run later against a process a reconnect spawns."""
    client = await spawn(_IGNORES_EOF)
    cleaned_up = []

    async def reader_still_running():
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned_up.append(client._proc)

    client._reader_task = asyncio.create_task(reader_still_running())
    await asyncio.sleep(0)
    proc = client._proc

    await client.aclose()

    assert cleaned_up == [proc], "the cleanup ran, against the process closed"
    assert client._proc is None


@pytest.mark.asyncio
async def test_a_respawn_lets_the_old_read_loop_clean_up_first(
    spawn, short_grace, monkeypatch, tmp_path
):
    """Run later, that cleanup would fail and forget the replacement."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    client = await spawn(_IGNORES_EOF)
    replaced = client._proc
    replacement = (await spawn(_IGNORES_EOF))._proc
    cleaned_up = []

    async def old_reader():
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned_up.append(client._proc)

    client._reader_task = asyncio.create_task(old_reader())
    await asyncio.sleep(0)

    async def create_subprocess_exec(*_args, **_kwargs):
        return replacement

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    await client._spawn()
    for task in (client._reader_task, client._stderr_task):
        task.cancel()

    assert cleaned_up == [replaced], "the old cleanup ran, against the old process"
    assert client._proc is replacement
    assert client._unexited_procs == [replaced]


@pytest.mark.asyncio
async def test_a_second_wait_for_the_read_loop_does_not_cut_its_cleanup_short():
    """A caller cancelled while waiting leaves the cleanup running; a retry waits."""
    client = CodexAppServerClient(binary="codex-under-test")
    cleanup_may_finish = asyncio.Event()
    cleaned_up = []

    async def reader():
        try:
            await asyncio.Event().wait()
        finally:
            await cleanup_may_finish.wait()
            cleaned_up.append(True)

    client._reader_task = asyncio.create_task(reader())
    await asyncio.sleep(0)
    first = asyncio.create_task(client._finish_io_tasks())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    second = asyncio.create_task(client._finish_io_tasks())
    await asyncio.sleep(0)
    cleanup_may_finish.set()
    await asyncio.wait_for(second, timeout=5.0)

    assert cleaned_up == [True]


@pytest.mark.asyncio
async def test_an_exited_process_is_not_kept_when_forgotten(spawn):
    client = await spawn(_EXITS_ON_EOF)
    proc = client._proc
    proc.stdin.close()
    await proc.wait()

    client._forget_proc()

    assert client._unexited_procs == []


@pytest.mark.asyncio
async def test_the_adapter_keeps_its_client_until_the_client_closes(
    spawn, short_grace, monkeypatch
):
    client = await spawn(_IGNORES_EOF)
    proc = client._proc
    real_kill = proc.kill
    adapter = CodexAdapter()
    adapter._client = client

    def refuse_kill():
        raise PermissionError("injected kill failure")

    monkeypatch.setattr(proc, "kill", refuse_kill)
    with pytest.raises(PermissionError):
        await adapter.aclose()

    assert adapter._client is client

    monkeypatch.setattr(proc, "kill", real_kill)
    await adapter.aclose()

    assert adapter._client is None
    assert proc.returncode is not None
