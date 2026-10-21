"""Retained failed-spawn worker reports its own terminal originals.

Real local PG failures and SQLite effects occur inside the retained worker.
Declared rollback/registry adapters are not authority or secure-deletion proofs.
"""
import asyncio

import pytest

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from kestrel_sovereign.multi_agent import agent_manager as manager_module
from kestrel_sovereign.multi_agent.config import LocalAgentConfig
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


class UnexpectedControl(BaseException):
    pass


def admission_for(manager, child):
    manager._pending_spawns = 1
    return manager_module.AgentOperationAdmission(
        name="RetainedChild", canonical_name="retainedchild", kind="spawn",
        registration_epoch=manager._agent_registration_shutdown_epoch,
        owner_task=asyncio.current_task(), child=child, spawn_slot_active=True,
        spawn_slot_terminal=asyncio.get_running_loop().create_future(),
    )


@pytest.fixture
async def retained_worker(onboarding_owner):
    manager, child, _ = onboarding_owner
    workers, callers = [], []
    async def start(admission):
        loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
        def factory(loop, coroutine, **kwargs):
            operation = _actual_operation(coroutine)
            task = previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
            if getattr(getattr(operation, "cr_code", None), "co_name", "") == "revoke_then_shutdown":
                workers.append(task)
            return task
        loop.set_task_factory(factory)
        try:
            caller = ownership.create_owned_task(manager._handoff_failed_spawn_cleanup(admission, child), retain_originals=True)
            callers.append(caller)
            outcome = await ownership.await_owned_task(caller)
            assert outcome.error is None and outcome.result is False
            assert len(workers) == 1, "retained cleanup acquired replacement workers"
            return workers[0]
        finally:
            loop.set_task_factory(previous)
    try:
        yield start
    finally:
        for task in workers:
            if not task.done():
                task.cancel("retire this fixture's retained worker only")
            await ownership.await_owned_task(task)
        for task in callers:
            await ownership.await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["receipt", "roster", "runtime", "witness"])
@pytest.mark.parametrize("later_outcome", ["would-succeed", "would-fail"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_retained_worker_native_failure_is_not_a_successful_retry(
    native_pg, onboarding_owner, retained_worker, monkeypatch, phase, later_outcome, source_cancel,
):
    manager, child, db = onboarding_owner
    admission = admission_for(manager, child)
    if phase == "receipt":
        admission.spawn_receipt_graph = object()  # Declared adapter, no signed graph claim.
    if phase == "roster":
        admission.spawn_startup_config = LocalAgentConfig(data_dir="retained", port=8801)
    if phase == "witness":
        admission.spawn_authority_witness_did = child.did
    sources, calls = [], []
    async def selected():
        calls.append(phase)
        await db.execute("INSERT INTO host_rollback_effects VALUES (?)", (420+len(calls),))
        if len(calls) == 1:
            # The actual failed PG transaction runs IN this original worker.
            source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
            sources.append(source)
            raise source.error
        if later_outcome == "would-fail":
            raise UnexpectedControl("later independent cleanup failure")
        admission.spawn_receipt_graph = None
        admission.spawn_startup_config = admission.spawn_startup_config_path = None
        admission.spawn_authority_witness_did = admission.spawn_authority_witness_mandate = None
        return False
    async def ordinary_runtime(*args):
        return False
    async def receipt(*args):
        return await selected()
    async def roster(*args):
        return await selected()
    async def runtime(*args):
        return await selected()
    async def witness(*args):
        return await selected()
    monkeypatch.setattr(manager, "_rollback_uncommitted_spawn_runtime", runtime if phase == "runtime" else ordinary_runtime)
    if phase == "receipt":
        monkeypatch.setattr(manager, "_downgrade_uncommitted_spawn_receipt", receipt)
    if phase == "roster":
        monkeypatch.setattr(manager, "_withdraw_uncommitted_spawn_startup_registration", roster)
    if phase == "witness":
        monkeypatch.setattr(manager, "_withdraw_uncommitted_spawn_authority_witness", witness)
    worker = await retained_worker(admission)
    result = await ownership.await_owned_task(worker)
    assert len(sources) == 1
    sources[0].assert_retained(result.error)
    assert calls == [phase] * (1 if source_cancel else 2)
    await asyncio.sleep(0)
    sources[0].assert_retained(manager._quarantined_shutdown_control)
    assert worker.done() and not manager._quarantined_shutdown_reapers
    assert manager._quarantined_cleanup_name_is_reserved("retainedchild")
    incomplete = source_cancel or later_outcome == "would-fail"
    assert manager._pending_spawns == int(incomplete)
    assert admission.spawn_slot_terminal.done() is (not incomplete)
    assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == len(calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["receipt", "roster", "runtime", "witness"])
async def test_retained_worker_ordinary_availability_still_retries(
    onboarding_owner, retained_worker, monkeypatch, phase,
):
    manager, child, db = onboarding_owner
    admission = admission_for(manager, child)
    if phase == "receipt":
        admission.spawn_receipt_graph = object()
    if phase == "roster":
        admission.spawn_startup_config = LocalAgentConfig(data_dir="ordinary", port=8801)
    if phase == "witness":
        admission.spawn_authority_witness_did = child.did
    calls = []
    async def selected(*args):
        calls.append(phase)
        await db.execute("INSERT INTO host_rollback_effects VALUES (?)", (430+len(calls),))
        if len(calls) == 1:
            raise OSError("ordinary availability failure")
        admission.spawn_receipt_graph = None
        admission.spawn_startup_config = admission.spawn_startup_config_path = None
        admission.spawn_authority_witness_did = admission.spawn_authority_witness_mandate = None
        return False
    async def runtime(*args):
        return False
    monkeypatch.setattr(manager, "_rollback_uncommitted_spawn_runtime", selected if phase == "runtime" else runtime)
    target = {"receipt": "_downgrade_uncommitted_spawn_receipt",
              "roster": "_withdraw_uncommitted_spawn_startup_registration",
              "witness": "_withdraw_uncommitted_spawn_authority_witness"}.get(phase)
    if target:
        monkeypatch.setattr(manager, target, selected)
    worker = await retained_worker(admission)
    result = await ownership.await_owned_task(worker)
    assert result.error is None and calls == [phase, phase]
    assert manager._pending_spawns == 0 and admission.spawn_slot_terminal.done()
    assert vars(manager).get("_quarantined_shutdown_control") is None
    assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("later", ["success", "witness-failure", "witness-withdrawal", "slot-failure"])
async def test_retained_worker_keeps_actual_runtime_withdrawal_through_remaining_settlement(
    onboarding_owner, retained_worker, monkeypatch, later,
):
    manager, child, db = onboarding_owner
    admission = admission_for(manager, child)
    admission.unpublished_cleanup_deferred_to_spawn = True
    admission.spawn_authority_witness_did = child.did
    entered, release, witness_entered, witness_release = (asyncio.Event() for _ in range(4))
    original_phase, observations, witness_observations = [], [], []
    later_failure = OriginalFailure.capture(UnexpectedControl("original later settlement failure"))
    worker = None
    async def discard(*args):
        original_phase.append(asyncio.current_task())
        entered.set()
        await release.wait()
        await db.execute("INSERT INTO host_rollback_effects VALUES (441)")
    async def offboard(*args, **kwargs):
        return False, None
    async def refund(*args, **kwargs):
        return False
    async def witness(*args):
        await db.execute("INSERT INTO host_rollback_effects VALUES (442)")
        if later == "witness-failure":
            raise later_failure.error
        if later == "witness-withdrawal":
            witness_entered.set()
            try:
                await witness_release.wait()
            except asyncio.CancelledError as error:
                witness_observations.append(OriginalFailure.capture(error))
                raise
        admission.spawn_authority_witness_did = admission.spawn_authority_witness_mandate = None
    monkeypatch.setattr(manager, "_discard_unpublished_initialized_agent", discard)
    monkeypatch.setattr(manager, "_offboard_agent_runtime_namespace", offboard)
    monkeypatch.setattr(manager, "_release_child_budget_cancellation_safe", refund)
    monkeypatch.setattr(manager, "record_expired_spawn_retirement", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_withdraw_uncommitted_spawn_authority_witness", witness)
    actual_resolve = manager._resolve_spawn_slot_terminal
    def resolve(current):
        actual_resolve(current)
        if later == "slot-failure":
            raise later_failure.error
    monkeypatch.setattr(manager, "_resolve_spawn_slot_terminal", resolve)
    actual_owned = ownership.await_owned_task
    async def observe(task, *args, **kwargs):
        outcome = await actual_owned(task, *args, **kwargs)
        if asyncio.current_task() is worker and task in original_phase and outcome.cancellation is not None:
            observations.append(OriginalFailure.capture(outcome.cancellation))
        return outcome
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    monkeypatch.setattr(manager_module, "await_owned_task", observe)
    worker = await retained_worker(admission)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        worker.cancel("actual retained runtime observer withdrew")
        await asyncio.sleep(0)
        release.set()
        if later == "witness-withdrawal":
            await asyncio.wait_for(witness_entered.wait(), 5)
            worker.cancel("actual later witness observer withdrew")
        result = await actual_owned(worker)
        assert len(observations) == 1
        observations[0].assert_retained(result.error)
        if later in {"witness-failure", "slot-failure"}:
            later_failure.assert_retained(result.error)
        if later == "witness-withdrawal":
            assert len(witness_observations) == 1
            witness_observations[0].assert_retained(result.error)
        assert all(task.done() for task in original_phase) and len(original_phase) == 1
        assert not execution_cleanup_debts(result.error), "retired phase acquired invented native debt"
        assert vars(manager).get("_quarantined_shutdown_control") is None, "finite caller withdrawal poisoned shared native history"
        assert manager._pending_spawns == int(later.startswith("witness-"))
        assert admission.spawn_slot_terminal.done() is (not later.startswith("witness-"))
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 2
    finally:
        release.set()
        witness_release.set()
        await actual_owned(worker)
        for task in original_phase:
            await actual_owned(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_retained_worker_returned_native_carrier_is_not_retirement_success(
    native_pg, onboarding_owner, retained_worker, monkeypatch, source_cancel,
):
    manager, child, db = onboarding_owner
    admission = admission_for(manager, child)
    sources = []
    async def runtime(*args):
        await db.execute("INSERT INTO host_rollback_effects VALUES (451)")
        source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
        sources.append(source)
        return source.error
    monkeypatch.setattr(manager, "_rollback_uncommitted_spawn_runtime", runtime)
    worker = await retained_worker(admission)
    result = await ownership.await_owned_task(worker)
    assert len(sources) == 1
    sources[0].assert_retained(result.error)
    assert manager._pending_spawns == 0 and admission.spawn_slot_terminal.done()
    await asyncio.sleep(0)
    sources[0].assert_retained(manager._quarantined_shutdown_control)
    assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1


@pytest.mark.asyncio
async def test_retained_worker_native_prefix_survives_actual_retry_withdrawal(
    native_pg, onboarding_owner, retained_worker, monkeypatch,
):
    from kestrel_sovereign.execution_custody import _explicit_control_chain
    manager, child, db = onboarding_owner
    admission = admission_for(manager, child)
    entered, sources, observations = asyncio.Event(), [], []
    worker = None
    async def runtime(*args):
        await db.execute("INSERT INTO host_rollback_effects VALUES (461)")
        source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, False))
        sources.append(source)
        raise source.error
    actual_sleep = asyncio.sleep
    async def sleep(delay, *args, **kwargs):
        selected = asyncio.current_task() is worker and delay == 1.0
        if selected:
            entered.set()
        try:
            return await actual_sleep(delay, *args, **kwargs)
        except asyncio.CancelledError as error:
            if selected:
                observations.append(OriginalFailure.capture(error))
            raise
    monkeypatch.setattr(manager, "_rollback_uncommitted_spawn_runtime", runtime)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    worker = await retained_worker(admission)
    await asyncio.wait_for(entered.wait(), 5)
    worker.cancel("actual retained restrictive retry observer withdrew")
    result = await ownership.await_owned_task(worker)
    assert len(sources) == len(observations) == 1
    sources[0].assert_retained(result.error)
    observations[0].assert_retained(result.error)
    sources[0].assert_retained(manager._quarantined_shutdown_control)
    assert observations[0].error not in tuple(_explicit_control_chain(manager._quarantined_shutdown_control))
    assert manager._pending_spawns == 1 and not admission.spawn_slot_terminal.done()
    assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
