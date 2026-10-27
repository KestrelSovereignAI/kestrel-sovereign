"""Driver-owned child cancellation is distinct from its original caller's Stop.

Actual registered tool/HTTP/SQLite ownership, with bounded pre-effect work. These
cases make no Docker/model call and never fabricate a worker action receipt.
"""
import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import Request
from fastapi.routing import APIRoute
from kestrel_sdk.resources import ResourceNotFound, ResourceOutcomeUnknown
from starlette.responses import JSONResponse

from kestrel_sovereign._async_ownership import await_owned_task, raise_owned_outcome
from kestrel_sovereign.auth import caller_context_scope
from kestrel_sovereign.endpoints.agent_helpers import get_caller
from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
from kestrel_sovereign.visual_computer.docker import DockerComputerDriver
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.unit.visual_computer.test_host import AUTH, intent, path
from tests.unit.visual_computer.test_host import (
    visual_host as visual_host,  # noqa: PLC0414 - pytest fixture re-export
)


@pytest_asyncio.fixture
async def borrowed(visual_host, harness, monkeypatch, tmp_path):
    client, agents, app = visual_host
    feature = agents['kite'].features['VisualComputerFeature']
    response = await client.post(path() + '/requests', headers=AUTH, json=intent())
    assert response.status_code == 200, response.text
    session = response.json()
    sid = session['session_id']
    observed = await client.get(path() + f'/sessions/{sid}/observation', headers=AUTH)
    assert observed.status_code == 200
    payload = {'action_id': 'original-action', 'session_id': sid,
        'observation_id': observed.json()['frame']['observation_id'], 'control_epoch': 0,
        'not_after': (datetime.now(UTC) + timedelta(seconds=20)).isoformat(),
        'action': {'kind': 'click', 'x': 1, 'y': 1}}
    # Only constructor metadata and the genuine owned borrowing function run.
    driver = DockerComputerDriver(docker_socket=str(tmp_path / 'never-opened.sock'),
                                  egress_guard=feature.runtime.egress.operation)
    entered, left = asyncio.Event(), asyncio.Event()
    blocker = asyncio.Event()
    state = SimpleNamespace(client=client, app=app, feature=feature, h=harness,
        driver=driver, session=session, sid=sid, payload=payload, entered=entered,
        left=left, blocker=blocker, facts={'entries': 0}, cleanup=None)
    original_act = harness.adapter.act
    async def act(session, request):
        owner = asyncio.current_task()
        state.owner_task = owner
        async def operation(current):
            state.facts['entries'] += 1
            state.borrower = asyncio.current_task()
            current()
            entered.set()
            try:
                await blocker.wait()
                return await original_act(session, request)
            finally:
                try:
                    if state.cleanup is not None:
                        await state.cleanup()
                finally:
                    left.set()
        try:
            return await driver._guarded_call(session, request.action_id, driver.egress_guard, operation)
        except BaseException as error:
            state.facts['owner_cancelling_at_error'] = owner.cancelling()
            state.facts['original_error'] = error
            raise
    async def action_status(session, action_id):
        if action_id not in harness.adapter.receipts:
            raise ResourceNotFound('bounded worker has no original receipt')
        return harness.adapter.receipts[action_id]
    monkeypatch.setattr(harness.adapter, 'act', act)
    monkeypatch.setattr(harness.adapter, 'action_status', action_status)
    tool, = [t for t in feature.get_tools() if t.name == 'visual_computer_action']
    async def registered_tool(request: Request):
        # The normal middleware creates this actual authenticated object. It is
        # neither constructed nor restored from a frozen caller DTO here.
        caller = get_caller(request)
        assert caller is request.state.caller and caller.is_sovereign
        with caller_context_scope(caller):
            result = await tool.execute(request_json=(await request.body()).decode())
        return JSONResponse(result)
    route = APIRoute('/api/visual-computer/registered-tool-probe', registered_tool, methods=['POST'])
    app.router.routes.append(route)
    state.route = route
    try:
        yield state
    finally:
        app.router.routes.remove(route)
        assert not driver._borrowers
        assert not feature.runtime.egress._calls


async def stop(b):
    return await b.client.post(path() + f'/sessions/{b.sid}/stop', headers=AUTH,
        json={'request_id': 'original-stop', 'session_id': b.sid,
              'expected_control_epoch': 0, 'operation': 'pause'})


def submit(b, boundary):
    target = path() + (f'/sessions/{b.sid}/actions' if boundary == 'http' else '/registered-tool-probe')
    return asyncio.create_task(b.client.post(target, headers=AUTH, json=b.payload))


async def assert_unknown_original(b):
    status = await b.client.get(path() + f'/sessions/{b.sid}/actions/original-action', headers=AUTH)
    assert status.status_code == 503
    assert status.json()['error']['code'] == 'visual_outcome_unknown'
    duplicate = await b.client.post(path() + f'/sessions/{b.sid}/actions', headers=AUTH, json=b.payload)
    assert duplicate.status_code == 503
    assert b.facts['entries'] == 1 and b.h.adapter.effects == 0 and b.h.adapter.receipts == {}
    assert b.feature.runtime._inflight == 0 and b.left.is_set()
    assert b.borrower.done() and not b.driver._borrowers
    row = await b.h.db.fetchone('SELECT pending FROM visual_computer_actions WHERE action_id = ?', ('original-action',))
    assert row == (1,) or tuple(row) == (1,)


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['http', 'registered-tool'])
async def test_session_stop_is_unknown_without_cancelling_original_caller(borrowed, boundary):
    b = borrowed
    active = submit(b, boundary)
    try:
        await asyncio.wait_for(b.entered.wait(), 5)
        paused = await asyncio.wait_for(stop(b), 5)
        assert paused.status_code == 200 and paused.json()['stopped'] is True
        response = await asyncio.wait_for(active, 5)
        assert b.facts['owner_cancelling_at_error'] == 0
        assert active.cancelled() is False
        assert isinstance(b.facts['original_error'], ResourceOutcomeUnknown)
        if boundary == 'http':
            assert response.status_code == 503, response.text
            assert response.json()['error']['code'] == 'visual_outcome_unknown'
            assert response.headers['cache-control'].startswith('private, no-store')
        else:
            assert response.status_code == 200, response.text
            assert response.json() == {'success': False, 'error': 'original_visual_operation_interrupted',
                                       'tool': 'visual_computer_action'}
        await assert_unknown_original(b)
    finally:
        if not active.done():
            active.cancel()
        await asyncio.gather(active, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['http', 'registered-tool'])
async def test_genuine_caller_cancellation_remains_original_control(borrowed, boundary):
    b = borrowed
    active = submit(b, boundary)
    try:
        await asyncio.wait_for(b.entered.wait(), 5)
        assert active.cancel('original-client-withdrawal')
        with pytest.raises(asyncio.CancelledError):
            await active
        assert b.facts['owner_cancelling_at_error'] > 0
        assert isinstance(b.facts['original_error'], asyncio.CancelledError)
        assert active.cancelled()
        # AnyIO can uncancel the outer client's terminal counter. The actual
        # delivered original ACT error/cancellation above is the observation.
        await assert_unknown_original(b)
    finally:
        if not active.done():
            active.cancel()
        await asyncio.gather(active, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['http', 'registered-tool'])
async def test_stop_and_repeated_caller_cancellation_hold_exact_cleanup_and_map(borrowed, boundary):
    b = borrowed
    closing, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    finalizers = []
    async def finalizer():
        closing.set()
        await release.wait()
        cleaned.set()
    async def cleanup():
        task = asyncio.create_task(finalizer())
        finalizers.append(task)
        outcome = await await_owned_task(task)
        raise_owned_outcome(outcome, operation='original bounded borrower cleanup')
    b.cleanup = cleanup
    active = submit(b, boundary)
    paused = None
    try:
        await asyncio.wait_for(b.entered.wait(), 5)
        paused = asyncio.create_task(stop(b))
        await asyncio.wait_for(closing.wait(), 5)
        assert b.driver._borrowers[b.sid] is b.borrower
        assert not b.borrower.done() and not active.done() and not paused.done()
        active.cancel('first-original-client-withdrawal')
        await asyncio.sleep(0)
        active.cancel('second-original-client-withdrawal')
        await asyncio.sleep(0)
        assert not cleaned.is_set() and not finalizers[0].done()
        assert b.driver._borrowers[b.sid] is b.borrower
        assert not b.borrower.done() and not active.done() and not paused.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await active
        result = await asyncio.wait_for(paused, 5)
        assert result.status_code == 200 and result.json()['stopped'] is True
        assert cleaned.is_set() and finalizers[0].done()
        await assert_unknown_original(b)
    finally:
        release.set()
        if not active.done():
            active.cancel()
        await asyncio.gather(active, *([] if paused is None else [paused]), *finalizers, return_exceptions=True)


class NativeAbort(BaseException):
    pass


class BorrowControl(asyncio.CancelledError):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['cause', 'group', 'commit', 'retirement', 'cancel-retirement', 'cancel-subclass', 'ordinary-cause', 'base-control'])
async def test_borrower_preserves_original_explicit_control_graph(harness, tmp_path, kind):
    h = harness
    ctx = h.context()
    marker = NativeAbort('original-native-control')
    if kind == 'cause':
        error = asyncio.CancelledError('original-child')
        error.__cause__ = marker
    elif kind == 'ordinary-cause':
        error = asyncio.CancelledError('original child observation')
        error.__cause__ = ResourceOutcomeUnknown('original native transport unknown')
    elif kind == 'group':
        error = BaseExceptionGroup('original-controls', [asyncio.CancelledError('child'), marker])
    elif kind == 'commit':
        error = ExecutionCommitOutcomeError('unknown')
    elif kind == 'retirement':
        error = RuntimeError('original-retirement-debt')
        error._kestrel_native_retirement_debts = ((marker, None),)
    elif kind == 'cancel-retirement':
        error = asyncio.CancelledError('original-retirement-control')
        error._kestrel_native_retirement_debts = ((marker, None),)
    elif kind == 'cancel-subclass':
        error = BorrowControl('original-specific-child-control')
    else:
        error = marker
    with h.bind(ctx):
        session = await h.runtime.admit(ctx, h.request(ctx))
        driver = DockerComputerDriver(docker_socket=str(tmp_path / 'never-opened.sock'),
                                      egress_guard=h.runtime.egress.operation)
        async def origin():
            async with (
                h.runtime.operation(ctx, VisualAction.ACT) as check,
                h.runtime.egress.activate(session, 'graph-action', check,
                    deadline=datetime.now(UTC) + timedelta(seconds=10)),
            ):
                async def operation(current):
                    current()
                    raise error
                return await driver._guarded_call(session, 'graph-action', driver.egress_guard, operation)
        with pytest.raises(BaseException) as caught:
            await origin()
        assert caught.value is error
        if kind == 'cause':
            assert caught.value.__cause__ is marker
        assert driver._borrowers == {} and h.runtime.egress._calls == {}


@pytest.mark.asyncio
@pytest.mark.parametrize('clear_counter', [False, True], ids=['retained-counter', 'cleared-counter'])
async def test_direct_owner_cancellation_is_not_hidden_by_transport_uncancel(harness, tmp_path, clear_counter):
    h = harness
    ctx = h.context()
    entered = asyncio.Event()
    with h.bind(ctx):
        session = await h.runtime.admit(ctx, h.request(ctx))
        driver = DockerComputerDriver(docker_socket=str(tmp_path / 'never-opened.sock'),
                                      egress_guard=h.runtime.egress.operation)
        async def origin():
            async with (
                h.runtime.operation(ctx, VisualAction.ACT) as check,
                h.runtime.egress.activate(session, 'caller-action', check,
                    deadline=datetime.now(UTC) + timedelta(seconds=10)),
            ):
                async def operation(current):
                    current()
                    entered.set()
                    await asyncio.Event().wait()
                return await driver._guarded_call(session, 'caller-action', driver.egress_guard, operation)
        active = asyncio.create_task(origin())
        await asyncio.wait_for(entered.wait(), 5)
        active.cancel('exact-original-owner-cancellation')
        if clear_counter:
            active.uncancel()
            assert active.cancelling() == 0
        with pytest.raises(asyncio.CancelledError) as caught:
            await active
        assert caught.value.args == ('exact-original-owner-cancellation',)
        assert driver._borrowers == {} and h.runtime.egress._calls == {}


@pytest.mark.asyncio
async def test_original_inline_sqlite_commit_cancellation_is_not_availability(
        harness, tmp_path, monkeypatch):
    """An actual native signal COMMIT can retain facts on a plain cancellation.

    This is the canonical inline-SQLite representation, not a fabricated
    acknowledgment or caller-assigned commit metadata.
    """
    from kestrel_sovereign.execution_custody import (
        execution_commit_outcome,
        execution_terminal_error,
        is_execution_control_error,
    )
    from tests.unit.test_durable_signal_delivery import _channel_dispatcher, _close
    from tests.unit.test_signals_dispatcher import _action_reg, _signal

    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / 'original-commit.db', 'did:agent:original-borrow-commit')
    journals = []
    original_write = dispatcher._write_outcome_log
    async def observe_original_writer(*args, **kwargs):
        journals.append(dispatcher._original_outcome_writer.get())
        return await original_write(*args, **kwargs)
    monkeypatch.setattr(dispatcher, '_write_outcome_log', observe_original_writer)
    dispatcher._registry.register(_action_reg('original.borrow.commit'))
    signal = _signal('original.borrow.commit')
    try:
        await dispatcher.dispatch_signal(signal)
        assert await backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 1
        journal, = journals
        assert journal.has_acknowledged_commit() and journal._phase == 'acknowledged'
        original = asyncio.CancelledError('original inline COMMIT observation')
        retained = journal.retain_committed(original)
        assert retained is original and original.__cause__ is None
        assert not is_execution_control_error(original)
        assert execution_commit_outcome(original) == 'committed'
        selected = execution_terminal_error(original, RuntimeError('later ordinary noise'))
        assert selected is original and selected.__cause__ is None

        h = harness
        ctx = h.context()
        with h.bind(ctx):
            session = await h.runtime.admit(ctx, h.request(ctx))
            driver = DockerComputerDriver(docker_socket=str(tmp_path / 'never-opened.sock'),
                                          egress_guard=h.runtime.egress.operation)
            async with (
                h.runtime.operation(ctx, VisualAction.ACT) as check,
                h.runtime.egress.activate(session, 'original-commit-action', check,
                    deadline=datetime.now(UTC) + timedelta(seconds=10)),
            ):
                async def operation(current):
                    current()
                    raise selected
                with pytest.raises(BaseException) as caught:
                    await driver._guarded_call(session, 'original-commit-action', driver.egress_guard, operation)
            assert driver._borrowers == {} and h.runtime.egress._calls == {}
            assert caught.value is original
            assert execution_commit_outcome(caught.value) == 'committed'
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)
