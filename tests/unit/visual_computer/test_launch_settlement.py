"""Original native launch facts require real exec/reap and exact gated owners."""
import asyncio
import errno
import sys
from types import SimpleNamespace

import pytest
from pydantic import SecretStr
from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
from kestrel_sovereign.visual_computer.dispatch import original_dispatch, native_dispatch
from kestrel_sovereign.visual_computer.docker import DockerBoundary, DockerComputerDriver, PROVIDER
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.unit.test_visual_native_dispatch_fact import originals
from tests.unit.visual_computer.test_runtime import opened


def executable(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(f'#!{sys.executable}\n' + body)
    path.chmod(0o700)
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize('code', [errno.ENOENT, errno.EACCES, errno.ENOEXEC])
@pytest.mark.parametrize('phase', ['preflight', 'rpc'])
async def test_real_exec_refusal_settles_original_action_and_dedup(harness, monkeypatch, tmp_path, code, phase):
    h = harness
    ctx = h.context()
    h.adapter.driver_id = PROVIDER
    target = tmp_path / 'owned-exec-refusal'
    if code != errno.ENOENT:
        target.write_text('owned non-program\n')
        target.chmod(0o600 if code == errno.EACCES else 0o700)
    errors = []
    with h.bind(ctx):
        session, request = await opened(h, ctx)
        driver = DockerComputerDriver(docker_socket=str(tmp_path/'not-contacted.sock'),
            egress_guard=h.runtime.egress.operation)
        driver.boundary.executable = str(target)
        original = driver.boundary.invoke
        async def native(*args, **kwargs):
            try:
                return await original(*args, **kwargs)
            except OSError as error:
                errors.append(error)
                raise
        monkeypatch.setattr(driver.boundary, 'invoke', native)
        driver._connections[session.session_id] = SimpleNamespace(credential=SecretStr('a'*64))
        async def bound(*args, **kwargs):
            if phase == 'preflight':
                return await driver.boundary.invoke(['container', 'inspect', 'c'*64])
            return {'container_id':'c'*64}
        monkeypatch.setattr(driver, '_bound', bound)
        h.runtime.drivers = {PROVIDER:driver}
        with pytest.raises(OSError) as caught:
            await h.runtime.act(ctx, request)
        assert caught.value is errors[0] and caught.value.errno == code
        assert len(errors) == 1
        assert not await h.repository.has_pending_actions(h.runtime.scope(ctx), session.session_id)
        receipt = await h.runtime.action_status(ctx, session.session_id, request.action_id)
        assert receipt.action_id == request.action_id and receipt.outcome == 'denied'
        assert receipt.reason_code == 'host_not_dispatched'
        assert await h.runtime.act(ctx, request) == receipt
        assert len(errors) == 1, 'dedup must not repeat the original native attempt'
        assert (await h.runtime.release(ctx, session.session_id))['released']
        assert await h.repository.records(h.runtime.scope(ctx)) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('at', ['command', 'launch', 'stream'])
@pytest.mark.parametrize('eager', [False, True] if hasattr(asyncio, 'eager_task_factory') else [False])
async def test_factory_accept_then_raise_keeps_payload_gated_and_exact_owners_joined(monkeypatch, tmp_path, at, eager):
    marker = tmp_path/'stdin-effect'
    program = executable(tmp_path, 'owned-native',
        f'import sys\nfrom pathlib import Path\ndata=sys.stdin.buffer.read()\n'
        f'if data:Path({str(marker)!r}).write_bytes(data)\nprint("{{}}")\n')
    boundary = DockerBoundary(str(tmp_path/'not-contacted.sock'))
    boundary.executable = str(program)
    children, accepted = [], []
    actual = asyncio.create_subprocess_exec
    async def capture(*args, **kwargs):
        process = await actual(*args, **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', capture)
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    refusal = RuntimeError('owned factory failed after acceptance')
    names = {'command':'_invoke', 'launch':'launch_process', 'stream':'write'}
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        options = {'eager_start':True} if eager else {}
        task = asyncio.Task(coroutine, loop=loop, **kwargs, **options)
        if getattr(getattr(operation, 'cr_code', None), 'co_name', None) == names[at]:
            accepted.append(task)
            raise refusal
        return task
    session, request = originals()
    try:
        loop.set_task_factory(factory)
        with original_dispatch(session, request) as origin:
            with native_dispatch(session, request, PROVIDER) as trace:
                with pytest.raises(BaseException) as caught:
                    await boundary.invoke(['owned'], data=b'private-input', _input_trace=trace)
            assert refusal in tuple(_explicit_control_chain(caught.value))
            assert trace.no_input(), 'actual accepted failed gate and child must be joined'
            assert origin.trace.refusal(caught.value) is not None
        assert accepted and all(task.done() for task in accepted)
        assert len(children) == (1 if at == 'stream' else 0)
        assert all(process.returncode is not None for process in children)
        assert not marker.exists()
    finally:
        loop.set_task_factory(previous)
        for task in accepted:
            await asyncio.gather(task, return_exceptions=True)
        for process in children:
            if process.returncode is None:
                process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_launch_result_publication_failure_reaps_retained_exact_process(monkeypatch, tmp_path):
    marker = tmp_path/'stdin-effect'
    program = executable(tmp_path, 'owned-native',
        f'import sys\nfrom pathlib import Path\ndata=sys.stdin.buffer.read()\n'
        f'if data:Path({str(marker)!r}).write_bytes(data)\n')
    boundary = DockerBoundary(str(tmp_path/'not-contacted.sock'))
    boundary.executable = str(program)
    actual = asyncio.create_subprocess_exec
    children, accepted = [], []
    async def capture(*args, **kwargs):
        child = await actual(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', capture)
    refusal = RuntimeError('actual launch result publication failed')
    class ResultFailure(asyncio.Task):
        def result(self):
            super().result()
            raise refusal
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(getattr(operation, 'cr_code', None), 'co_name', None) == 'launch_process':
            task = ResultFailure(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return asyncio.Task(coroutine, loop=loop, **kwargs)
    try:
        loop.set_task_factory(factory)
        with pytest.raises(RuntimeError) as caught:
            await boundary.invoke(['owned'], data=b'private-input')
        assert caught.value is refusal
        assert len(children) == len(accepted) == 1
        assert children[0].returncode is not None and accepted[0].done()
        assert not marker.exists()
    finally:
        loop.set_task_factory(previous)
        for process in children:
            if process.returncode is None:
                process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_possible_partial_stdin_remains_unknown_after_exact_reap(monkeypatch, tmp_path):
    marker = tmp_path/'first-byte'
    ready = tmp_path/'reader-ready'
    program = executable(tmp_path, 'owned-reader',
        f'import sys,time\nfrom pathlib import Path\nPath({str(ready)!r}).touch()\nbyte=sys.stdin.buffer.read(1)\n'
        f'Path({str(marker)!r}).write_bytes(byte)\ntime.sleep(60)\n')
    boundary = DockerBoundary(str(tmp_path/'not-contacted.sock'))
    boundary.executable = str(program)
    actual = asyncio.create_subprocess_exec
    children = []
    async def capture(*args, **kwargs):
        process = await actual(*args, **kwargs)
        children.append(process)
        async with asyncio.timeout(2):
            while not ready.exists():
                await asyncio.sleep(.005)
        return process
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', capture)
    session, request = originals()
    with original_dispatch(session, request) as origin:
        with native_dispatch(session, request, PROVIDER) as trace:
            with pytest.raises(ResourceOutcomeUnknown) as caught:
                await boundary.invoke(['owned'], data=b'X'*(4*1024*1024), timeout=.5, _input_trace=trace)
        assert marker.exists(), (children[0].returncode, trace.input_started,
            getattr(caught.value, 'diagnostics', None))
        assert marker.read_bytes() == b'X'
        assert trace.input_started and not trace.no_input()
        assert origin.trace.refusal(RuntimeError('not a no-effect proof')) is None
    assert len(children) == 1 and children[0].returncode is not None


@pytest.mark.asyncio
async def test_repeated_cancellation_joins_accepted_launch_without_stdin(monkeypatch, tmp_path):
    marker = tmp_path/'stdin-effect'
    program = executable(tmp_path, 'owned-reader',
        f'import sys\nfrom pathlib import Path\ndata=sys.stdin.buffer.read()\n'
        f'if data:Path({str(marker)!r}).write_bytes(data)\n')
    boundary = DockerBoundary(str(tmp_path/'not-contacted.sock'))
    boundary.executable = str(program)
    actual = asyncio.create_subprocess_exec
    entered, release = asyncio.Event(), asyncio.Event()
    children = []
    async def capture(*args, **kwargs):
        process = await actual(*args, **kwargs)
        children.append(process)
        entered.set()
        await release.wait()
        return process
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', capture)
    task = asyncio.create_task(boundary.invoke(['owned'], data=b'private-input'))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        for _ in range(4):
            task.cancel('original caller cancelled repeatedly')
            await asyncio.sleep(0)
        assert not task.done() and children[0].returncode is None
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert children[0].returncode is not None and not marker.exists()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        for process in children:
            if process.returncode is None:
                process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_foreign_launch_error_after_acceptance_is_unknown_not_exec_refusal(monkeypatch, tmp_path):
    program = executable(tmp_path, 'owned-reader', 'import time\ntime.sleep(60)\n')
    boundary = DockerBoundary(str(tmp_path/'not-contacted.sock'))
    boundary.executable = str(program)
    actual = asyncio.create_subprocess_exec
    children = []
    refusal = FileNotFoundError(errno.ENOENT, 'owned wrapper failed after launch', str(program))
    async def withhold(*args, **kwargs):
        children.append(await actual(*args, **kwargs))
        raise refusal
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', withhold)
    session, request = originals()
    try:
        with original_dispatch(session, request) as origin:
            with native_dispatch(session, request, PROVIDER) as trace:
                with pytest.raises(FileNotFoundError) as caught:
                    await boundary.invoke(['owned'], _input_trace=trace)
            assert caught.value is refusal
            assert trace.commands == 1 and not trace.no_input()
            assert origin.trace.refusal(refusal) is None
        assert len(children) == 1 and children[0].returncode is None
    finally:
        # Only this fixture knows its withheld native child. The boundary
        # correctly did not claim to settle an undelivered Process.
        for child in children:
            if child.returncode is None:
                child.kill()
            await child.wait()


@pytest.mark.asyncio
async def test_opaque_task_factory_keeps_explicit_native_dispatch_debt(tmp_path):
    boundary = DockerBoundary(str(tmp_path/'not-contacted.sock'))
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    accepted = []
    refusal = RuntimeError('owned opaque factory refused publication')
    def factory(loop, coroutine, **kwargs):
        async def outer():
            return await coroutine
        accepted.append(asyncio.Task(outer(), loop=loop, **kwargs))
        raise refusal
    session, request = originals()
    try:
        loop.set_task_factory(factory)
        with original_dispatch(session, request) as origin:
            with native_dispatch(session, request, PROVIDER) as trace:
                with pytest.raises(BaseException) as caught:
                    await boundary.invoke(['owned'], _input_trace=trace)
            assert refusal in tuple(_explicit_control_chain(caught.value))
            assert execution_cleanup_debts(caught.value)
            assert trace.commands == 1 and origin.trace.refusal(caught.value) is None
    finally:
        loop.set_task_factory(previous)
        for task in accepted:
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stdin_close_error_still_physically_reaps_original_child(monkeypatch, tmp_path):
    program = executable(tmp_path, 'owned-reader', 'import sys,time\nsys.stdin.buffer.read()\ntime.sleep(60)\n')
    boundary = DockerBoundary(str(tmp_path/'not-contacted.sock'))
    boundary.executable = str(program)
    actual, children, closers = asyncio.create_subprocess_exec, [], []
    refusal = RuntimeError('original native stdin close failed')
    async def capture(*args, **kwargs):
        child = await actual(*args, **kwargs)
        children.append(child)
        closers.append(child.stdin.close)
        def refused_close():
            raise refusal
        child.stdin.close = refused_close
        return child
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', capture)
    try:
        with pytest.raises(BaseException) as caught:
            await boundary.invoke(['owned'])
        assert refusal in _explicit_control_chain(caught.value)
        assert len(children) == 1 and children[0].returncode is not None
    finally:
        for child, close in zip(children, closers):
            child.stdin.close = close
            close()
            if child.returncode is None:
                child.kill()
            await child.wait()


@pytest.mark.asyncio
async def test_reaper_admission_failure_retains_exact_live_process_debt(monkeypatch, tmp_path):
    program = executable(tmp_path, 'owned-reader', 'import sys,time\nsys.stdin.buffer.read()\ntime.sleep(60)\n')
    boundary = DockerBoundary(str(tmp_path/'not-contacted.sock'))
    boundary.executable = str(program)
    actual, children, accepted = asyncio.create_subprocess_exec, [], []
    async def capture(*args, **kwargs):
        child = await actual(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', capture)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refusal = RuntimeError('actual native reaper publication failed')
    def factory(loop, coroutine, **kwargs):
        task = asyncio.Task(coroutine, loop=loop, **kwargs)
        operation = _actual_operation(coroutine)
        if getattr(getattr(operation, 'cr_code', None), 'co_name', None) == 'retire':
            accepted.append(task)
            raise refusal
        return task
    def current():
        raise RuntimeError('original prewrite authority refused')
    try:
        loop.set_task_factory(factory)
        with pytest.raises(BaseException) as caught:
            await boundary.invoke(['owned'], current=current)
        assert refusal in _explicit_control_chain(caught.value)
        assert len(children) == len(accepted) == 1 and accepted[0].done()
        assert children[0].returncode is None, 'failed start gate did not admit the reaper'
        assert any(resource is children[0] for resource, _ in execution_cleanup_debts(caught.value)), (
            'a joined failed gate must not erase the original live Process debt')
    finally:
        loop.set_task_factory(previous)
        for task in accepted:
            await asyncio.gather(task, return_exceptions=True)
        for child in children:
            if child.returncode is None:
                child.kill()
            await child.wait()
