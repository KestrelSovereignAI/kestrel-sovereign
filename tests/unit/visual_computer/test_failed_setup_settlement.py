"""Close exact original setup custody only after durable settlement.

Real Host/SQLite and actual original Tasks/Futures; daemon/transport and physical
retirement bodies are explicitly SCRIPTED. These are not native Docker ACKs.
"""
from datetime import timedelta
from uuid import uuid4

import pytest
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown, ResourceUnsupported
from kestrel_sovereign._async_ownership import await_owned_task, raise_owned_outcome
from kestrel_sovereign.visual_computer.local_worker import _LocalWorkerResources
from kestrel_sovereign.visual_computer.repository import CustodyRecord
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.unit.visual_computer.test_local_two_domain_admit import local_setup as local_setup
from tests.unit.visual_computer.test_two_domain_setup_publication import scripted_daemon_maps as scripted_daemon_maps
from tests.unit.visual_computer.test_two_domain_worker_profile import (
    measured_created_browser_profile as measured_created_browser_profile,
    measured_created_broker_profile as measured_created_broker_profile,
)
from tests.unit.visual_computer.test_setup_cleanup_sweeps import controlled_setup as controlled_setup, leaves


async def sql_flags(repository, scope, request_id):
    row = await repository.db.fetchone(
        "SELECT revision, released, retired FROM visual_computer_requests WHERE owner = ? AND request_id = ?",
        (repository._owner(scope), request_id),
    )
    return list(row)


async def join_original_cleanup(resource):
    task = resource._close_admission.task
    assert task is resource._original_close_admission.task
    assert getattr(task, "_kestrel_owned_admission", None) is resource._close_admission
    outcome = await await_owned_task(task)
    raise_owned_outcome(outcome, operation="test original physical cleanup", retain_originals=True)
    assert await resource.original._finished is None


async def assert_capacity_available(setup, context):
    request = setup.request(context, request_id="capacity-" + uuid4().hex)
    async with setup.runtime.operation(context, VisualAction.ADMIT) as current:
        record = CustodyRecord(request=request, session_id=uuid4().hex, last_activity_at=setup.runtime._now())
        await setup.runtime.repository.admit(request.scope, record, current)
        await setup.runtime.repository.retire_unacquired(request.scope, record, at=setup.runtime._now(), guard=current)


@pytest.mark.asyncio
@pytest.mark.parametrize("persisted_lease", [False, True], ids=["no-accepted-lease", "accepted-lease-before-open"])
async def test_failed_setup_settles_only_actual_durable_lease_facts(scripted_daemon_maps, monkeypatch, persisted_lease):
    setup, state = scripted_daemon_maps
    repository = setup.runtime.repository
    repository.max_requests = 1
    refusal = OSError("SCRIPTED original opening-save refusal")
    original_save = repository.save
    observed = []

    async def refuse_opening(scope, record, guard=None):
        if record.phase == "opening":
            current = await repository.get(scope, request_id=record.request.request_id, guard=guard)
            observed.append(current)
            assert current.phase == "leased" and current.lease is not None
            assert current.connection is current.session is None
            raise refusal
        return await original_save(scope, record, guard)

    state["failure"] = None if persisted_lease else "internal-network"
    if persisted_lease:
        monkeypatch.setattr(repository, "save", refuse_opening)
    async with setup.bound() as context:
        request = setup.request(context, request_id="settle-original-" + str(persisted_lease))
        error = None
        try:
            await setup.runtime.admit(context, request)
        except BaseException as original:
            error = original
        assert error is refusal if persisted_lease else isinstance(error, ResourceUnsupported)
        assert bool(observed) is persisted_lease
        resource = state["resources"]
        await join_original_cleanup(resource)
        assert state["closed"] == [resource]
        current = await repository.get(request.scope, request_id=request.request_id)
        flags = await sql_flags(repository, request.scope, request.request_id)
        assert current.session is current.connection is None
        if persisted_lease:
            assert current.phase == "released" and current.lease.state == "released"
            assert current.lease.lease_id == observed[0].lease.lease_id and flags[1:] == [1, 0]
        else:
            assert current.phase == "retired" and current.lease is None
            assert current.retired_from == "acquiring" and flags[1:] == [0, 1]
        assert setup.runtime.egress._calls == {} and setup.driver._connections == {}
        reconciled = await setup.runtime.reconcile_request(context, request.request_id)
        released = await setup.runtime.release(context, current.session_id)
        assert reconciled["released"] is released["released"] is True
        if not persisted_lease:
            assert reconciled["acquired"] is released["acquired"] is None
            assert released["setup_cleanup"] == "verified_original_physical_retirement"
            with pytest.raises(ResourceConflict, match="retired after verified setup cleanup"):
                await setup.runtime.admit(context, request)  # Original ID never recreates.
        assert state["closed"] == [resource]  # No replacement/re-close program.
        await assert_capacity_available(setup, context)


@pytest.mark.asyncio
async def test_revoked_setup_settlement_retains_original_until_fresh_reconcile(scripted_daemon_maps, monkeypatch):
    setup, state = scripted_daemon_maps
    state["failure"] = "internal-network"
    repository = setup.runtime.repository
    repository.max_requests = 1
    original_clock = setup.runtime.authority.clock
    original_retire = _LocalWorkerResources._retire
    expired_at = None

    async def retire_then_expire(resource):
        await original_retire(resource)
        monkeypatch.setattr(setup.runtime.authority, "clock", lambda: expired_at)

    monkeypatch.setattr(_LocalWorkerResources, "_retire", retire_then_expire)
    async with setup.bound() as context:
        expired_at = context.expires_at + timedelta(seconds=1)
        request = setup.request(context, request_id="settlement-authority-expired")
        error = None
        try:
            await setup.runtime.admit(context, request)
        except BaseException as original:
            error = original
        assert error is not None
        assert any(isinstance(leaf, OperatorAuthorizationError) for leaf in leaves(error))
    monkeypatch.setattr(setup.runtime.authority, "clock", original_clock)
    resource = state["resources"]
    await join_original_cleanup(resource)
    current = await repository.get(request.scope, request_id=request.request_id)
    assert current.phase == "acquiring" and (await sql_flags(repository, request.scope, request.request_id))[1:] == [0, 0]
    assert setup.runtime.egress._calls.get(resource.original._key) is resource.original
    async with setup.bound() as fresh:
        result = await setup.runtime.reconcile_request(fresh, request.request_id)
        assert result["released"] is True and result["acquired"] is None
        await assert_capacity_available(setup, fresh)
    assert state["closed"] == [resource] and setup.runtime.egress._calls == {}


@pytest.mark.asyncio
async def test_competing_original_row_write_cannot_discard_setup_cleanup(scripted_daemon_maps, monkeypatch):
    setup, state = scripted_daemon_maps
    state["failure"] = "internal-network"
    repository = setup.runtime.repository
    original_retire = repository._retire_failed_local_setup
    competing = []

    async def advance_original(scope, record, original, *, at, guard):
        current = await repository.get(scope, request_id=record.request.request_id, guard=guard)
        current.last_activity_at = current.last_activity_at + timedelta(microseconds=1)
        await repository.save(scope, current, guard)
        competing.append(current.revision)
        return await original_retire(scope, record, original, at=at, guard=guard)

    monkeypatch.setattr(repository, "_retire_failed_local_setup", advance_original)
    async with setup.bound() as context:
        request = setup.request(context, request_id="settlement-current-revision")
        error = None
        try:
            await setup.runtime.admit(context, request)
        except BaseException as original:
            error = original
        assert error is not None
        assert any(isinstance(leaf, ResourceConflict) and "changed concurrently" in str(leaf) for leaf in leaves(error))
    assert len(competing) == 1
    resource = state["resources"]
    await join_original_cleanup(resource)
    current = await repository.get(request.scope, request_id=request.request_id)
    assert current.revision == competing[0] and current.phase == "acquiring"
    assert setup.runtime.egress._calls.get(resource.original._key) is resource.original
    assert (await sql_flags(repository, request.scope, request.request_id))[1:] == [0, 0]
    monkeypatch.setattr(repository, "_retire_failed_local_setup", original_retire)
    async with setup.bound() as fresh:
        result = await setup.runtime.reconcile_request(fresh, request.request_id)
        assert result["released"] is True and result["acquired"] is None
    assert state["closed"] == [resource] and setup.runtime.egress._calls == {}


@pytest.mark.asyncio
async def test_unknown_original_retirement_cannot_settle_or_prune(controlled_setup, monkeypatch):
    setup, state = controlled_setup
    setup_error = ResourceUnsupported("SCRIPTED failed setup before controller")
    cleanup_error = ResourceOutcomeUnknown("SCRIPTED unresolved original retirement")
    state["known_debt"].extend((setup_error, cleanup_error))

    async def fail_bootstrap(resource, request, credential):
        raise setup_error

    async def unknown_retirement(resource):
        raise cleanup_error

    monkeypatch.setattr(_LocalWorkerResources, "bootstrap", fail_bootstrap)
    monkeypatch.setattr(_LocalWorkerResources, "_retire", unknown_retirement)
    async with setup.bound() as context:
        request = setup.request(context, request_id="unknown-cleanup-keeps-custody")
        error = None
        try:
            await setup.runtime.admit(context, request)
        except BaseException as original:
            error = original
        assert error is not None
        assert any(leaf is cleanup_error for leaf in leaves(error))
        resource = state["resources"][0]
        outcome = await await_owned_task(resource._close_admission.task)
        assert outcome.error is cleanup_error
        finished = await await_owned_task(resource.original._finished)
        assert finished.error is cleanup_error
        assert resource._closed is False
        current = await setup.runtime.repository.get(request.scope, request_id=request.request_id)
        assert current.phase != "released"
        assert (await sql_flags(setup.runtime.repository, request.scope, request.request_id))[1:] == [0, 0]
        assert setup.runtime.egress._calls.get(resource.original._key) is resource.original
        assert setup.driver._connections.get(current.session_id) is resource
        later = None
        try:
            await setup.runtime.release(context, current.session_id)
        except BaseException as original:
            later = original
        assert any(leaf is cleanup_error for leaf in leaves(later))
        assert setup.runtime.egress._calls.get(resource.original._key) is resource.original
        assert setup.driver._connections.get(current.session_id) is resource


@pytest.mark.asyncio
@pytest.mark.parametrize("refuse_release_write", [False, True], ids=["ACK", "write-refusal-then-ACK"])
async def test_local_connection_prunes_only_after_durable_release(controlled_setup, monkeypatch, refuse_release_write):
    setup, state = controlled_setup
    original_save = setup.runtime.repository.save
    refusal = OSError("SCRIPTED released-row storage refusal")

    async def refuse(scope, record, guard=None):
        if record.phase == "released":
            assert state["resources"][0]._closed
            raise refusal
        return await original_save(scope, record, guard)

    async with setup.bound() as context:
        session = await setup.runtime.admit(context, setup.request(
            context, request_id="release-prune-" + str(refuse_release_write),
            required_capabilities=frozenset({"screenshot"})))
    resource = state["resources"][0]
    assert setup.driver._connections.get(session.session_id) is resource
    if refuse_release_write:
        monkeypatch.setattr(setup.runtime.repository, "save", refuse)
        async with setup.bound() as context:
            with pytest.raises(OSError) as observed:
                await setup.runtime.release(context, session.session_id)
        assert observed.value is refusal
        await join_original_cleanup(resource)
        current = await setup.runtime.repository.get(session.scope, session_id=session.session_id)
        assert current.phase == "releasing" and current.lease.state != "released"
        assert setup.driver._connections.get(session.session_id) is resource
        monkeypatch.setattr(setup.runtime.repository, "save", original_save)
    async with setup.bound() as fresh:
        result = await setup.runtime.release(fresh, session.session_id)
    assert result["released"] is True
    await join_original_cleanup(resource)
    current = await setup.runtime.repository.get(session.scope, session_id=session.session_id)
    assert current.phase == "released" and current.lease.state == "released"
    assert session.session_id not in setup.driver._connections
