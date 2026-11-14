"""Real Host/SQLite cleanup interleavings; native transport remains scripted.

The installed canonical controller and original owned retirement Tasks execute.
No Docker, browser, TLS, cloud or physical-retirement conformance is claimed.
"""
import asyncio
from datetime import timedelta
from types import SimpleNamespace

import pytest
from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sovereign._async_ownership import OwnedTaskAdmission, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.visual_computer.local_worker import _LocalWorkerResources
from tests.unit.visual_computer.test_local_two_domain_admit import local_setup as local_setup
from tests.unit.visual_computer.test_two_domain_lifecycle import (
    admit_original,
    lifecycle_transport as lifecycle_transport,
)
from tests.unit.visual_computer.test_two_domain_retirement import (
    _join_release_tasks,
    _raise_release_errors,
    _release_task_boundary,
)


async def failed_setup_with_retained_settlement(setup, state, monkeypatch):
    """Keep a genuine failed original after a deliberately unavailable SQL save."""
    bootstrap = _LocalWorkerResources.bootstrap
    save = setup.runtime.repository.save
    open_error = ResourceOutcomeUnknown("scripted_local_bootstrap_reply_unavailable")
    write_error = OSError("scripted_original_retirement_settlement_unavailable")
    writes = {"remaining_failures": 1, "refused": 0}

    async def failed_bootstrap(resource, request, credential):
        await bootstrap(resource, request, credential)
        # The original driver already retains the exact connection and the
        # repository has ACKed its lease/opening row before bootstrap begins.
        raise open_error

    async def unavailable_settlement(scope, record, check=None):
        if record.phase == "released" and writes["remaining_failures"]:
            writes["remaining_failures"] -= 1
            writes["refused"] += 1
            raise write_error
        return await save(scope, record, check)

    monkeypatch.setattr(_LocalWorkerResources, "bootstrap", failed_bootstrap)
    monkeypatch.setattr(setup.runtime.repository, "save", unavailable_settlement)
    async with setup.bound() as context:
        now = setup.runtime._now()
        request = setup.request(context, expires_at=now + timedelta(seconds=30),
                                idle_ttl_seconds=20, required_capabilities=frozenset({"screenshot"}))
        with pytest.raises(ResourceOutcomeUnknown) as caught:
            await setup.runtime.admit(context, request)
    assert caught.value is open_error
    assert any(error is write_error for error in _explicit_control_chain(caught.value))
    resource, = state["resources"]
    original = resource.original
    record = await setup.runtime.repository.get(request.scope, request_id=request.request_id)
    assert record.phase == "opening" and record.lease is not None and record.connection is not None
    assert record.session is None and resource._closed and writes["refused"] == 1
    assert setup.runtime.egress._calls[original._key] is original
    assert setup.driver._connections[record.session_id] is resource
    # A closed flag is descriptive. Join and authenticate the SAME completed
    # original Future and retirement Task before proceeding with the test.
    await original._join_retirement(record, failed=True)
    return SimpleNamespace(record=record, resource=resource, original=original,
                           writes=writes, write_error=write_error)


@pytest.mark.asyncio
async def test_expired_failed_setup_maintenance_prunes_only_after_original_join_and_durable_ack(
    lifecycle_transport, monkeypatch,
):
    setup, state = lifecycle_transport
    retained = await failed_setup_with_retained_settlement(setup, state, monkeypatch)
    original_join = retained.original._join_retirement
    joins = []

    async def observe_original_join(record, *, failed):
        joins.append((record.revision, failed, setup.runtime._lock(record.session_id).locked()))
        return await original_join(record, failed=failed)

    monkeypatch.setattr(retained.original, "_join_retirement", observe_original_join)
    with monkeypatch.context() as clock:
        clock.setattr(setup.runtime.authority, "clock", lambda: retained.record.lease.expires_at + timedelta(microseconds=1))
        uncertain = await setup.runtime.maintain_owned_sessions(retained.record.request.scope)
    saved = await setup.runtime.repository.get(retained.record.request.scope, session_id=retained.record.session_id)
    assert uncertain == [] and saved.phase == "released" and saved.lease.state == "released"
    assert saved.connection is None and saved.session is None
    assert setup.runtime.egress._calls == {}, "durably released failed setup still retains original egress custody"
    assert setup.driver._connections == {}
    assert joins == [(retained.record.revision, True, True)], "maintenance must join the retained failed setup under the session lock"
    assert state["page_closed"] == 1


@pytest.mark.asyncio
async def test_maintenance_failed_durable_settlement_keeps_same_original_for_next_sweep(
    lifecycle_transport, monkeypatch,
):
    setup, state = lifecycle_transport
    retained = await failed_setup_with_retained_settlement(setup, state, monkeypatch)
    retained.writes["remaining_failures"] = 1
    with monkeypatch.context() as clock:
        clock.setattr(setup.runtime.authority, "clock", lambda: retained.record.lease.expires_at + timedelta(microseconds=1))
        uncertain = await setup.runtime.maintain_owned_sessions(retained.record.request.scope)
        pending = await setup.runtime.repository.get(retained.record.request.scope, session_id=retained.record.session_id)
        assert uncertain == [retained.record.session_id] and pending.phase != "released"
        assert setup.runtime.egress._calls[retained.original._key] is retained.original
        assert setup.driver._connections[pending.session_id] is retained.resource
        assert retained.writes["refused"] == 2
        await retained.original._join_retirement(pending, failed=True)
        assert await setup.runtime.maintain_owned_sessions(pending.request.scope) == []
    saved = await setup.runtime.repository.get(retained.record.request.scope, session_id=retained.record.session_id)
    assert saved.phase == "released" and saved.connection is None
    assert setup.runtime.egress._calls == {} and setup.driver._connections == {}
    assert state["page_closed"] == 1


@pytest.mark.asyncio
async def test_maintenance_does_not_settle_unexpired_nonidle_failed_setup(lifecycle_transport, monkeypatch):
    setup, state = lifecycle_transport
    retained = await failed_setup_with_retained_settlement(setup, state, monkeypatch)
    with monkeypatch.context() as clock:
        clock.setattr(setup.runtime.authority, "clock", lambda: retained.record.request.requested_at + timedelta(seconds=1))
        assert await setup.runtime.maintain_owned_sessions(retained.record.request.scope) == []
    saved = await setup.runtime.repository.get(retained.record.request.scope, session_id=retained.record.session_id)
    assert saved == retained.record
    assert setup.runtime.egress._calls[retained.original._key] is retained.original
    assert setup.driver._connections[saved.session_id] is retained.resource
    assert retained.writes["refused"] == 1


@pytest.mark.asyncio
async def test_reconcile_released_connection_discard_serializes_with_original_release(
    lifecycle_transport, monkeypatch,
):
    setup, state = lifecycle_transport
    session = await admit_original(setup)
    resource, = state["resources"]
    save = setup.runtime.repository.save
    lost_ack = OSError("scripted_released_row_ack_delivery_lost")
    acknowledged = []

    async def lose_release_delivery(scope, record, check=None):
        result = await save(scope, record, check)
        if record.phase == "released" and not acknowledged:
            acknowledged.append(record.revision)
            raise lost_ack
        return result

    monkeypatch.setattr(setup.runtime.repository, "save", lose_release_delivery)
    async with setup.bound() as context:
        with pytest.raises(OSError) as caught:
            await setup.runtime.release(context, session.session_id)
    assert caught.value is lost_ack
    monkeypatch.setattr(setup.runtime.repository, "save", save)
    saved = await setup.runtime.repository.get(session.scope, session_id=session.session_id)
    assert acknowledged and saved.phase == "released" and saved.connection is None
    assert setup.driver._connections[session.session_id] is resource
    assert setup.runtime.egress._calls == {}
    admissions = (OwnedTaskAdmission(), OwnedTaskAdmission())
    entered, second_entered, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    join = resource.original._join_retirement
    lock = setup.runtime._lock(session.session_id)
    calls = []

    async def held_original_join(record, *, failed):
        task = asyncio.current_task()
        calls.append((task, failed, lock.locked()))
        if task is admissions[0].task:
            entered.set()
            await finish.wait()
        elif task is admissions[1].task:
            second_entered.set()
        return await join(record, failed=failed)

    monkeypatch.setattr(resource.original, "_join_retirement", held_original_join)
    primary, boundary = None, None
    async with setup.bound() as context:
        try:
            create_owned_task(setup.runtime.reconcile_request(context, saved.request.request_id),
                              admission=admissions[0], name="original-released-reconcile")
            await asyncio.wait_for(entered.wait(), 5)
            create_owned_task(setup.runtime.release(context, session.session_id),
                              admission=admissions[1], name="original-concurrent-released-release")
            boundary = await _release_task_boundary(admissions[1].task, lock, second_entered)
        except BaseException as error:
            primary = error
        finally:
            finish.set()
            outcomes, join_errors = await _join_release_tasks(admissions)
    _raise_release_errors(primary, outcomes, join_errors)
    assert boundary == "same-original-lock-queued", "released-row discard escaped its existing session lock"
    assert [outcome.result for outcome in outcomes] == [{"session_id": session.session_id, "released": True}] * 2
    assert calls == [(admissions[0].task, False, True)]
    assert setup.driver._connections == {} and setup.runtime.egress._calls == {}
    assert state["page_closed"] == 1
