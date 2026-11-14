"""Lifecycle sweeps retain a failed session and continue healthy sessions.

The real runtime, original egress owner and SQLite ledger are exercised here.
Only the adapter and a controlled clock interleaving are test inputs; this is
not native worker retirement evidence.
"""
import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.resources import ResourceOutcomeUnknown

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import execution_terminal_error


def retained_interruption(harness, session, original, terminal):
    """Retain native-join-shaped errors on the actual revoked adapter task.

    The cancellation is delivered by the original egress owner. Authorization
    rejection comes from that same call's unchanged current-authority check.
    These controlled adapter cases do not claim native Docker retirement.
    """
    if terminal == "bare":
        return original
    if terminal == "cancelled-join":
        cause = asyncio.CancelledError("original nested launch join")
    else:
        call, = [call for key, call in harness.runtime.egress._calls.items()
                 if key[3] == session.session_id and key[-1] == "session"]
        clock = call.clock
        call.clock = lambda: call.deadline
        try:
            call.check()
        except OperatorAuthorizationError as error:
            cause = error
        else:
            pytest.fail("the revoked original egress call accepted more work")
        finally:
            call.clock = clock
        if terminal == "unknown-authorization":
            unknown = ResourceOutcomeUnknown("original transport interrupted")
            unknown.__cause__ = cause
            cause = unknown
    original.__cause__ = cause
    return execution_terminal_error(original, retain_originals=True)


async def two_sessions(harness):
    context = harness.context()
    sessions = []
    with harness.bind(context):
        for request_id in ("0-original-failed-pause", "1-original-healthy-pause"):
            sessions.append(await harness.runtime.admit(
                context, harness.request(context, request_id=request_id)))
    rows = await harness.repository.records(sessions[0].scope)
    assert [row.session_id for row in rows] == [s.session_id for s in sessions]
    return sessions


async def assert_failed_then_healthy(harness, sessions, uncertain):
    first, second = sessions
    failed = await harness.repository.get(first.scope, session_id=first.session_id)
    healthy = await harness.repository.get(second.scope, session_id=second.session_id)
    assert uncertain == [first.session_id]
    assert failed.phase == "fenced" and failed.pending_control is not None
    assert failed.session.controller == "agent"
    assert healthy.phase == "fenced" and healthy.pending_control is None
    assert healthy.session.controller == "none"
    assert healthy.session.control_epoch == second.control_epoch + 1
    assert harness.runtime.egress._calls == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["borrower", "caller"])
@pytest.mark.parametrize("terminal", ["bare", "cancelled-join", "authorization", "unknown-authorization"])
async def test_lifecycle_pause_distinguishes_borrower_revocation_from_caller_cancel(
        harness, monkeypatch, interruption, terminal):
    h = harness
    sessions = await two_sessions(h)
    first, second = sessions
    control = h.adapter.control
    entered = asyncio.Event()
    original_tasks = []

    async def held_first_control(session, request):
        if session.session_id == first.session_id:
            original_tasks.append(asyncio.current_task())
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as error:
                raise retained_interruption(h, first, error, terminal)
        return await control(session, request)

    monkeypatch.setattr(h.adapter, "control", held_first_control)
    caller = asyncio.create_task(h.runtime.fence_owned_sessions(first.scope))
    outcome = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        if interruption == "borrower":
            await h.runtime.egress.revoke_session(first.scope, first.session_id)
        else:
            caller.cancel("original-lifecycle-caller-withdrawal")
        outcome = await await_owned_task(caller)
        assert len(original_tasks) == 1 and original_tasks[0].done()
        if interruption == "borrower":
            assert not caller.cancelled(), "a revoked borrower cancelled the sweep owner"
            assert outcome.error is None and outcome.cancellation is None
            await assert_failed_then_healthy(h, sessions, outcome.result)
        else:
            assert caller.cancelled()
            assert type(outcome.error) is asyncio.CancelledError
            assert outcome.error.args == ("original-lifecycle-caller-withdrawal",)
            row = await h.repository.get(second.scope, session_id=second.session_id)
            assert row.phase == "active" and row.pending_control is None
            assert h.runtime.egress._calls == {}
    finally:
        if not caller.done():
            caller.cancel()
        await await_owned_task(caller)


@pytest.mark.asyncio
async def test_lifecycle_pause_expired_original_borrow_does_not_abort_later_session(
        harness, monkeypatch):
    h = harness
    sessions = await two_sessions(h)
    first, _second = sessions
    offset = timedelta()
    errors = []

    def clock():
        return datetime.now(UTC) + offset

    # Only this test's injected egress clock moves. The original _ActiveCall
    # checker, current-authority binding and durable ledger remain unchanged.
    monkeypatch.setattr(h.runtime.egress, "clock", clock)
    control = h.adapter.control

    async def first_borrow_expires(session, request):
        nonlocal offset
        if session.session_id == first.session_id:
            calls = [call for key, call in h.runtime.egress._calls.items()
                     if key[3] == first.session_id and key[-1] == "session"]
            call, = calls
            assert asyncio.current_task() in call.accepted_jobs
            offset = timedelta(hours=1)
            try:
                call.check()
            except OperatorAuthorizationError as error:
                errors.append(error)
                raise
            finally:
                offset = timedelta()
            pytest.fail("the original expired egress borrow was accepted")
        return await control(session, request)

    monkeypatch.setattr(h.adapter, "control", first_borrow_expires)
    uncertain = await h.runtime.fence_owned_sessions(first.scope)
    assert len(errors) == 1
    await assert_failed_then_healthy(h, sessions, uncertain)


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["borrower", "caller"])
@pytest.mark.parametrize("terminal", ["bare", "cancelled-join", "authorization", "unknown-authorization"])
async def test_foreground_observe_retains_its_original_caller_and_joined_cleanup(
        harness, monkeypatch, interruption, terminal):
    h = harness
    context = h.context()
    with h.bind(context):
        session = await h.runtime.admit(context, h.request(context))
        entered, cleanup_entered, finish_cleanup = (
            asyncio.Event(), asyncio.Event(), asyncio.Event())
        original_tasks, original_errors = [], []

        async def held_observe(original_session):
            assert original_session == session
            original_tasks.append(asyncio.current_task())
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as error:
                retained = retained_interruption(h, session, error, terminal)
                original_errors.append(retained)
                raise retained
            finally:
                cleanup_entered.set()
                await finish_cleanup.wait()

        monkeypatch.setattr(h.adapter, "observe", held_observe)
        caller = asyncio.create_task(h.runtime.observe(context, session.session_id))
        revoker = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if interruption == "borrower":
                revoker = asyncio.create_task(h.runtime.egress.revoke_session(
                    session.scope, session.session_id))
            else:
                caller.cancel("original-observe-caller-withdrawal")
            await asyncio.wait_for(cleanup_entered.wait(), 3)
            assert not caller.done() and h.runtime.egress._calls
            assert len(original_tasks) == 1 and not original_tasks[0].done()
            finish_cleanup.set()
            if revoker is not None:
                revocation = await await_owned_task(revoker)
                assert revocation.error is revocation.cancellation is None
            outcome = await await_owned_task(caller)
            assert len(original_errors) == 1 and original_tasks[0].done()
            if interruption == "borrower":
                assert not caller.cancelled(), "a revoked read borrower cancelled its caller"
                assert caller.cancelling() == 0 and outcome.cancellation is None
                assert type(outcome.error) is ResourceOutcomeUnknown
                assert str(outcome.error) == "original_visual_operation_interrupted"
                assert outcome.error.__cause__ is original_errors[0]
            else:
                assert caller.cancelled()
                assert type(outcome.error) is asyncio.CancelledError
                assert outcome.error.args == ("original-observe-caller-withdrawal",)
            saved = await h.repository.get(session.scope, session_id=session.session_id)
            assert saved.phase == "active" and saved.session == session
            assert saved.frame is None and saved.lease.state == "ready"
            assert h.runtime.egress._calls == {}
        finally:
            finish_cleanup.set()
            if not caller.done():
                caller.cancel()
            await await_owned_task(caller)
            if revoker is not None:
                await await_owned_task(revoker)


@pytest.mark.asyncio
async def test_foreground_observe_without_withdrawal_publishes_original_frame(harness):
    h = harness
    context = h.context()
    with h.bind(context):
        session = await h.runtime.admit(context, h.request(context))
        observation = await h.runtime.observe(context, session.session_id)
        saved = await h.repository.get(session.scope, session_id=session.session_id)
    assert saved.frame == observation.frame and observation.frame.session_id == session.session_id
    assert saved.phase == "active" and saved.session == session
    assert h.runtime.egress._calls == {}
