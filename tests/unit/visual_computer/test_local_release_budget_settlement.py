"""Real Host/SQLite release settlement after a controlled original close.

The installed canonical worker controller and original owned retirement task
execute. Only the existing lifecycle fixture's descriptive closer is delayed;
this is not Docker, browser, TLS or native-retirement conformance.
"""
import asyncio
import copy
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
from pydantic import SecretBytes

from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sovereign._async_ownership import OwnedTaskAdmission, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain

from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.unit.visual_computer.test_local_two_domain_admit import local_setup as local_setup
from tests.unit.visual_computer.test_two_domain_lifecycle import (
    admit_original,
    lifecycle_transport as lifecycle_transport,
)
from tests.unit.visual_computer.test_two_domain_retirement import _join_release_tasks


def retain_observation(name, value):
    directory = os.environ.get("CURRENT_ROW_PROOF_DIR")
    if directory:
        with (Path(directory) / ("RELEASE-BUDGET-" + name + ".json")).open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")


async def delayed_original_retirement(setup, resource, monkeypatch, *, after=None, event=None):
    original = resource._retire
    entered = []

    async def slower_original():
        entered.append(asyncio.current_task())
        if event is not None:
            event.set()
        # The real operation deadline expires while the original owned closer
        # remains alive. Its cancellation must never cancel this retained task.
        await asyncio.sleep(0.08)
        result = await original()
        if after is not None:
            after()
        return result

    monkeypatch.setattr(resource, "_retire", slower_original)
    setup.runtime.policy = replace(setup.runtime.policy, operation_seconds=0.01)
    return entered


async def original_completion_facts(setup, state, resource, session, entered, name):
    saved = await setup.runtime.repository.get(session.scope, session_id=session.session_id)
    original = resource.original
    admission = resource._close_admission
    task = admission.task
    assert admission is resource._original_close_admission
    assert getattr(task, "_kestrel_owned_admission", None) is admission
    assert entered == [task]
    # Actual awaits plus canonical original identity/configuration/lease/
    # connection checks are load-bearing; done()/closed alone are not proof.
    await original._join_retirement(saved, failed=False)
    assert original._finished is original._original_finished
    assert original._finished.result() is None and task.result() is None
    assert resource._closed and resource._closing and state["page_closed"] == 1
    value = {"phase": saved.phase, "lease_state": saved.lease.state,
        "released_at_present": saved.lease.released_at is not None,
        "connection_retained": saved.connection is not None,
        "driver_closer_retained": setup.driver._connections.get(session.session_id) is resource,
        "same_original_setup_future_joined": True,
        "same_original_owned_retirement_task_joined": True,
        "original_retirement_result_none": True,
        "controlled_page_closed_once": state["page_closed"] == 1,
        "Docker_browser_TLS_or_native_conformance": False}
    retain_observation(name, value)
    return saved


@pytest.mark.asyncio
async def test_release_timeout_preserved_but_current_grant_settles_original_retirement(
    lifecycle_transport, monkeypatch,
):
    setup, state = lifecycle_transport
    session = await admit_original(setup)
    resource, = state["resources"]
    policy = setup.runtime.policy
    try:
        async with setup.bound() as context:
            assert (context.expires_at - setup.runtime._now()).total_seconds() > 1
            entered = await delayed_original_retirement(setup, resource, monkeypatch)
            with pytest.raises(TimeoutError) as failure:
                await setup.runtime.release(context, session.session_id)
            assert type(failure.value) is TimeoutError
            # The policy deadline is not a fresh authority expiry. The exact
            # genuine current grant is still usable, without a guard=None path.
            async with setup.runtime.operation(context, VisualAction.STATUS) as check:
                check()
        saved = await original_completion_facts(setup, state, resource, session, entered, "foreground")
        assert saved.phase == "released" and saved.lease.state == "released", (
            "joined successful original retirement still charged after policy timeout")
        assert saved.connection is None and saved.lease.released_at is not None
        assert setup.driver._connections == {}
    finally:
        setup.runtime.policy = policy


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", ["grant", "configuration", "write", "revision"])
async def test_late_original_release_requires_current_guard_configuration_and_durable_cas(
    lifecycle_transport, monkeypatch, unavailable,
):
    setup, state = lifecycle_transport
    session = await admit_original(setup)
    resource, = state["resources"]
    policy, save, get = setup.runtime.policy, setup.runtime.repository.save, setup.runtime.repository.get
    attempts, changed = [], []
    write_error = OSError("controlled_late_original_release_write_refused")

    async def observe_original_save(scope, row, check=None):
        if row.phase == "released":
            attempts.append(check)
            assert check is not None, "foreground settlement bypassed its original grant"
            check()
            if unavailable == "write":
                raise write_error
        return await save(scope, row, check)

    async def independently_revise_current_row(scope, **kwargs):
        row = await get(scope, **kwargs)
        if unavailable == "revision" and resource._closed and not changed:
            assert kwargs["guard"] is not None
            row.last_activity_at += timedelta(microseconds=1)
            await save(scope, row, kwargs["guard"])
            changed.append(row.revision)
            return await get(scope, **kwargs)
        return row

    try:
        async with setup.bound() as context:
            with monkeypatch.context() as controlled:
                controlled.setattr(setup.runtime.repository, "save", observe_original_save)
                controlled.setattr(setup.runtime.repository, "get", independently_revise_current_row)

                def after_retirement():
                    if unavailable == "grant":
                        controlled.setattr(setup.runtime.authority, "clock",
                            lambda: context.expires_at + timedelta(microseconds=1))
                    elif unavailable == "configuration":
                        controlled.setattr(setup.provider, "_key", SecretBytes(b"x" * 32))

                entered = await delayed_original_retirement(setup, resource, controlled, after=after_retirement)
                with pytest.raises(TimeoutError) as failure:
                    await setup.runtime.release(context, session.session_id)
                assert type(failure.value) is TimeoutError
                if unavailable == "write":
                    assert write_error in list(_explicit_control_chain(failure.value))
        # Restore only the injected configuration/clock/write refusal; never
        # replace an original task or manufacture a completed retirement.
        saved = await original_completion_facts(setup, state, resource, session, entered, unavailable)
        assert saved.phase == "releasing" and saved.lease.state == "ready"
        assert saved.connection is not None and setup.driver._connections[session.session_id] is resource
        assert saved.lease.released_at is None
        assert len(attempts) == (1 if unavailable == "write" else 0)
        if unavailable == "revision":
            assert changed == [saved.revision]
    finally:
        setup.runtime.policy = policy


@pytest.mark.asyncio
async def test_unissued_original_closer_cannot_be_started_by_exception_settlement(
    lifecycle_transport, monkeypatch,
):
    setup, state = lifecycle_transport
    session = await admit_original(setup)
    resource, = state["resources"]
    denied = ResourceOutcomeUnknown("controlled_release_refused_before_closer_entry")

    async def refused_before_entry(scope, lease):
        raise denied

    with monkeypatch.context() as controlled:
        controlled.setattr(setup.provider, "release", refused_before_entry)
        async with setup.bound() as context:
            with pytest.raises(ResourceOutcomeUnknown) as failure:
                await setup.runtime.release(context, session.session_id)
            assert failure.value is denied
        saved = await setup.runtime.repository.get(session.scope, session_id=session.session_id)
        assert saved.phase == "releasing" and saved.lease.released_at is None
        assert resource._close_admission.task is None
        assert not resource._closed and not resource._closing and state["page_closed"] == 0
        assert setup.driver._connections[session.session_id] is resource


@pytest.mark.asyncio
async def test_original_release_receipt_rechecks_same_admission_after_its_actual_join(
    lifecycle_transport, monkeypatch,
):
    setup, state = lifecycle_transport
    session = await admit_original(setup)
    resource, = state["resources"]
    policy = setup.runtime.policy
    original_admission = resource._close_admission
    join = resource.original._join_retirement
    actual_joins = []

    async def changed_after_original_join(record, *, failed):
        await join(record, failed=failed)
        actual_joins.append(original_admission.task)
        # This is a changed descriptive holder, never another task, result,
        # admission grant or resource. Keep the real original for restoration.
        resource._close_admission = copy.copy(original_admission)

    try:
        async with setup.bound() as context:
            entered = await delayed_original_retirement(setup, resource, monkeypatch)
            with monkeypatch.context() as controlled:
                controlled.setattr(resource.original, "_join_retirement", changed_after_original_join)
                try:
                    with pytest.raises(TimeoutError):
                        await setup.runtime.release(context, session.session_id)
                finally:
                    resource._close_admission = original_admission
        saved = await original_completion_facts(setup, state, resource, session, entered, "postjoin-admission")
        assert actual_joins == [original_admission.task]
        assert saved.phase == "releasing" and saved.lease.released_at is None
        assert setup.driver._connections[session.session_id] is resource
    finally:
        setup.runtime.policy = policy


@pytest.mark.asyncio
async def test_failed_original_closer_never_supplies_a_release_receipt(local_setup, monkeypatch):
    # Own the same original lifecycle fixture's generator explicitly so its
    # expected repeated cleanup error is asserted, rather than hidden by a new
    # teardown kernel or allowed to become an unrelated pytest teardown error.
    controlled = lifecycle_transport.__wrapped__(local_setup, monkeypatch)
    setup, state = await anext(controlled)
    failure = ResourceOutcomeUnknown("controlled_original_retirement_remains_unknown")
    try:
        session = await admit_original(setup)
        resource, = state["resources"]
        retire = resource._retire

        async def failed_original():
            await retire()
            raise failure

        monkeypatch.setattr(resource, "_retire", failed_original)
        async with setup.bound() as context:
            with pytest.raises(ResourceOutcomeUnknown) as caught:
                await setup.runtime.release(context, session.session_id)
        assert caught.value is failure
        saved = await setup.runtime.repository.get(session.scope, session_id=session.session_id)
        assert saved.phase == "releasing" and saved.lease.released_at is None
        assert setup.driver._connections[session.session_id] is resource
        task = resource._close_admission.task
        assert getattr(task, "_kestrel_owned_admission", None) is resource._original_close_admission
        assert task.exception() is failure
        # A descriptive closed flag cannot override this exact failed task.
        assert resource._closed and state["page_closed"] == 1
    finally:
        with pytest.raises(ResourceOutcomeUnknown) as cleanup:
            await controlled.aclose()
        assert cleanup.value is failure


@pytest.mark.asyncio
async def test_caller_cancellation_escapes_after_current_original_release_settlement(
    lifecycle_transport, monkeypatch,
):
    setup, state = lifecycle_transport
    session = await admit_original(setup)
    resource, = state["resources"]
    policy = setup.runtime.policy
    entered_event = asyncio.Event()
    admission = OwnedTaskAdmission()
    outcomes = errors = None
    try:
        async with setup.bound() as context:
            entered = await delayed_original_retirement(setup, resource, monkeypatch, event=entered_event)
            # This case withdraws the actual caller before its policy deadline.
            setup.runtime.policy = policy
            try:
                create_owned_task(setup.runtime.release(context, session.session_id),
                    admission=admission, name="original-release-caller-cancellation")
                await asyncio.wait_for(entered_event.wait(), 5)
                admission.task.cancel()
            finally:
                outcomes, errors = await _join_release_tasks((admission,))
        assert errors == [] and len(outcomes) == 1
        assert any(isinstance(error, asyncio.CancelledError)
            for error in (outcomes[0].error, outcomes[0].cancellation) if error is not None)
        saved = await original_completion_facts(setup, state, resource, session, entered, "caller-cancellation")
        assert saved.phase == "released" and saved.lease.released_at is not None
        assert saved.connection is None and setup.driver._connections == {}
    finally:
        setup.runtime.policy = policy


@pytest.mark.asyncio
async def test_eligible_sweep_settles_original_retirement_despite_operation_deadline(
    lifecycle_transport, monkeypatch,
):
    setup, state = lifecycle_transport
    session = await admit_original(setup)
    resource, = state["resources"]
    # Existing durable releasing custody is independently eligible for the
    # housekeeping sweep; it mints no fresh user/input grant or acquisition.
    async with setup.bound() as context:
        async with setup.runtime.operation(context, VisualAction.RELEASE) as check:
            row = await setup.runtime.repository.get(session.scope, session_id=session.session_id, guard=check)
            await setup.runtime._begin_release(session.scope, row, check)
    policy = setup.runtime.policy
    try:
        entered = await delayed_original_retirement(setup, resource, monkeypatch)
        uncertain = await setup.runtime.maintain_owned_sessions(session.scope)
        saved = await original_completion_facts(setup, state, resource, session, entered, "maintenance")
        assert saved.phase == "released" and saved.lease.state == "released", (
            "eligible sweep left joined successful original retirement charged after deadline")
        assert uncertain == [], "durably settled original release must not remain sweep uncertainty"
        assert saved.connection is None and setup.driver._connections == {}
    finally:
        setup.runtime.policy = policy
