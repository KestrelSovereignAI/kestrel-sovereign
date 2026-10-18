"""Shared gather and synchronous owners retain exact original native receipts.

Budget callback uses a declared local SQLite effect, not paid funds acceptance.
SQLite drain starts with a real uncommitted statement; no fake successful ACK.
"""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task, gather_owned
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.native_admission_fault import NativeAdmissionFault
from tests.utils.original_failure import OriginalFailure

native_pg, onboarding_owner = _native_pg, _onboarding_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("resource_named", [False, True])
@pytest.mark.parametrize("cancel_siblings", [False, True])
async def test_gather_keeps_original_failed_gate_even_without_resource_keyword(
    native_pg, monkeypatch, mode, eager, resource_named, cancel_siblings,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    first, second = SQLiteBackend(":memory:"), SQLiteBackend(":memory:")
    await first.connect()
    await second.connect()
    first_connection, second_connection = first._connection, second._connection

    async def closer(backend):
        await backend.close()

    def selected(operation):
        frame = getattr(operation, "cr_frame", None)
        return getattr(operation, "cr_code", None) is closer.__code__ and frame.f_locals.get("backend") is first

    fault = NativeAdmissionFault(selected, source.error, mode, eager=eager)
    operation = gather_owned(
        (closer(backend) for backend in (first, second)),
        cancel_siblings_on_error=cancel_siblings,
        retirement_resource=first if resource_named else None,
        retain_originals=True, isolate_control_originals=True,
    )
    owner = create_owned_task(operation, retain_originals=True)
    try:
        with fault:
            result = await await_owned_task(owner)
        fault.assert_terminal_unstarted()
        source.assert_retained(result.error)
        if resource_named:
            assert any(value is first and task is fault.task for value, task in execution_cleanup_debts(result.error))
        assert first._connection is first_connection
        assert getattr(first_connection, "_thread", first_connection).is_alive()
        if not cancel_siblings:
            assert second._connection is None and not getattr(second_connection, "_thread", second_connection).is_alive()
    finally:
        await await_owned_task(owner)
        for task in fault.accepted:
            await await_owned_task(task)
        await first.close()
        await second.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_synchronous_budget_record_keeps_original_not_substitute_future(
    native_pg, onboarding_owner, monkeypatch, mode, eager, source_cancel,
):
    manager, _agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    resource = object()
    manager._child_budgets["OuterBudget"] = resource
    effects = []

    async def refund(_name):
        await db.execute("INSERT INTO host_rollback_effects VALUES (353)")
        effects.append(True)

    monkeypatch.setattr(manager, "_release_child_budget", refund)
    target = type(manager)._release_child_budget_cancellation_safe.__code__
    fault = NativeAdmissionFault(lambda operation: getattr(operation, "cr_code", None) is target, source.error, mode, eager=eager)
    record = None
    try:
        with fault:
            async with manager._lock:
                record = manager._start_child_budget_release("OuterBudget")
                if mode != "refuse":
                    assert record.task is fault.task
                    assert manager._inflight_removal_budget_releases[record.release_id] is record
            result = await await_owned_task(create_owned_task(
                manager._await_child_budget_release_cancellation_safe(record.task), retain_originals=True,
            ))
        fault.assert_terminal_unstarted()
        source.assert_retained(result.error)
        assert any(value is resource and task is fault.task for value, task in execution_cleanup_debts(result.error))
        assert manager._child_budgets["OuterBudget"] is resource and not effects
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=353") == 0
    finally:
        if record is not None:
            await await_owned_task(record.task)
        for task in fault.accepted:
            await await_owned_task(task)
        manager._child_budgets.pop("OuterBudget", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("eager", [False, True])
async def test_synchronous_sqlite_drain_retains_original_and_refuses_later_work(
    native_pg, tmp_path, monkeypatch, mode, eager,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    backend = SQLiteBackend(str(tmp_path / "actual-drain.db"))
    await backend.connect()
    connection = backend._connection
    await backend.execute("CREATE TABLE effects (id INTEGER)")
    await connection.execute("INSERT INTO effects VALUES (1)")
    assert connection.in_transaction
    fault = NativeAdmissionFault(
        lambda operation: getattr(operation, "cr_code", None) is SQLiteBackend._drain_cancelled_write.__code__,
        source.error, mode, eager=eager,
    )
    try:
        with fault:
            with pytest.raises(BaseException) as caught:
                backend._handoff_cancelled_write(connection, error=source.error)
        assert backend._cancelled_write_drain is fault.task
        if fault.task is not None:
            await await_owned_task(fault.task)
        fault.assert_terminal_unstarted()
        source.assert_retained(caught.value)
        assert any(value is connection and task is fault.task for value, task in execution_cleanup_debts(caught.value))
        with pytest.raises(BaseException) as later:
            await backend.execute("INSERT INTO effects VALUES (2)")
        source.assert_retained(later.value)
        assert connection.in_transaction, "failed drain gate executed a replacement rollback"
        async with connection.execute("SELECT id FROM effects") as cursor:
            assert [row[0] for row in await cursor.fetchall()] == [1]
    finally:
        for task in fault.accepted:
            await await_owned_task(task)
        await await_owned_task(create_owned_task(backend.close(), retain_originals=True))
        assert not getattr(connection, "_thread", connection).is_alive()
