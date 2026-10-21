"""Manager exceptional exits keep exact accepted owners and original graphs.

Real PostgreSQL sources, SQLite effects, worker-thread filesystem markers and
original Tasks. Declared topology/rollback seams confer no signed authority.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign.multi_agent.config import LocalAgentConfig
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.integration.test_native_completion_vs_observation import observation_factory
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


class UnexpectedControl(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_offboarding_done_worker_observer_failure_cannot_invent_pending_retirement(
    native_pg, onboarding_owner, monkeypatch, tmp_path, source_cancel,
):
    from kestrel_sovereign.features import isolated_runtime
    from kestrel_sovereign.execution_custody import _explicit_control_chain
    from kestrel_sovereign.multi_agent.agent_manager import RuntimeOffboardingRetainedError
    manager, agent, _ = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    marker = tmp_path / "physically-complete-original-worker"
    accepted, hits = [], []
    def remove(*args):
        marker.touch()
        return isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED
    monkeypatch.setattr(isolated_runtime, "remove_runtime_namespace", remove)
    class ObservationTask(asyncio.Task):
        def remove_done_callback(self, callback):
            result = super().remove_done_callback(callback)
            if self.done() and getattr(callback, "__name__", "") == "_on_completion":
                hits.append(self)
                raise source.error
            return result
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **kwargs):
        if getattr(getattr(_actual_operation(coroutine), "cr_code", None), "co_name", "") == "cleanup_and_retire":
            task = ObservationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    try:
        loop.set_task_factory(factory)
        record = manager._start_agent_runtime_offboarding_identity(name="Done", agent_id=agent.did)
        loop.set_task_factory(previous)
        _, failure = await manager._finish_agent_runtime_offboarding(record, cancellation_already_observed=False)
        source.assert_retained(failure)
        assert hits == accepted and marker.is_file() and record.task.done()
        assert not any(isinstance(error, RuntimeOffboardingRetainedError) and error.cleanup_pending
                       for error in _explicit_control_chain(failure)), "successful physical deletion was reclassified as pending by an observer"
    finally:
        loop.set_task_factory(previous)
        for task in accepted:
            await ownership.await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["registration", "cold", "loaded"])
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_registration_and_offboarding_creator_receipts_keep_unstarted_recipe_custody(
    native_pg, onboarding_owner, monkeypatch, surface, mode, eager, source_cancel,
):
    from tests.utils.native_admission_fault import NativeAdmissionFault
    from kestrel_sovereign.features import isolated_runtime
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    entries, pending = [], []
    async def register(*args):
        entries.append("registration")
        await db.execute("INSERT INTO host_rollback_effects VALUES (232)")
    def removal(*args):
        entries.append("deletion")
        return isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED
    monkeypatch.setattr(isolated_runtime, "remove_runtime_namespace", removal)
    monkeypatch.setattr(isolated_runtime, "remove_agent_runtime_namespace", removal)
    manager._scheduler_polling_managed_by_host = True
    manager._scheduler_tenant_registration_hook = register
    async def original_operation():
        if surface == "registration":
            return await manager._begin_dynamic_scheduler_tenant_registration(
                "Refused", agent.did, LocalAgentConfig(data_dir="refused", port=8801))
        if surface == "cold":
            await manager._admit_agent_runtime_offboarding_identity(name="Refused", agent_id=agent.did,
                config=None, revoked=None, pending_offboarding=pending)
            record = pending[0]
        else:
            record = manager._start_agent_runtime_offboarding(name="Refused", agent=agent)
        _, failure = await manager._finish_agent_runtime_offboarding(record, cancellation_already_observed=False)
        if failure is not None:
            raise failure
    def predicate(op):
        return (getattr(op, "cr_code", None) is register.__code__ if surface == "registration"
                else getattr(getattr(op, "cr_code", None), "co_name", "") == "cleanup_and_retire")
    fault = NativeAdmissionFault(predicate, source.error, mode, eager=eager)
    owner = ownership.create_owned_task(original_operation(), retain_originals=True)
    try:
        with fault:
            result = await ownership.await_owned_task(owner)
        fault.assert_terminal_unstarted()
        source.assert_retained(result.error)
        assert entries == []
        assert not manager._inflight_runtime_offboardings
        assert agent.did not in manager._scheduler_authority_by_did
        assert not manager.scheduler_lifecycle_lock(agent.did).locked()
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 0
    finally:
        await ownership.await_owned_task(owner)
        for task in fault.accepted:
            await ownership.await_owned_task(task)
        for coroutine in (*fault.operations, *fault.refused):
            if getattr(coroutine, "cr_frame", None) is not None and coroutine.cr_await is None:
                coroutine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_public_offboarding_keeps_native_source_without_inventing_observer_cancellation(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    from kestrel_sovereign.features import isolated_runtime
    from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_commit_outcome
    manager, agent, _ = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    calls = []
    def remove_namespace(*args):
        calls.append("original worker")
        raise source.error
    async def prepare(*args, **kwargs):
        return None
    async def serialized(name, *, pending_offboarding, **kwargs):
        pending_offboarding.append(manager._start_agent_runtime_offboarding_identity(name=name, agent_id=agent.did))
        return True
    monkeypatch.setattr(isolated_runtime, "remove_runtime_namespace", remove_namespace)
    monkeypatch.setattr(manager, "_begin_destructive_spawn_retirement", prepare)
    monkeypatch.setattr(manager, "_remove_agent_serialized", serialized)
    owner = ownership.create_owned_task(manager.remove_agent("Original", offboard_runtime=True), retain_originals=True)
    result = await ownership.await_owned_task(owner)
    source.assert_retained(result.error)
    assert execution_commit_outcome(result.error) == execution_commit_outcome(source.error)
    assert calls == ["original worker"] and not manager._inflight_runtime_offboardings
    cancellations = [error for error in _explicit_control_chain(result.error) if isinstance(error, asyncio.CancelledError)]
    assert cancellations == ([source.error] if source_cancel else []), "source cancellation was replaced/supplemented by an invented caller withdrawal"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["add", "remove"])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_wait_observer_failure_joins_exact_spawn_hook_before_exit(
    native_pg, onboarding_owner, monkeypatch, boundary, eager, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    unwind = OriginalFailure.capture(OSError("original accepted hook unwind failed"))
    entered, release, faulted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    accepted, fences = [], []
    async def hook():
        await db.execute("INSERT INTO host_rollback_effects VALUES (221)")
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
            await db.execute("INSERT INTO host_rollback_effects VALUES (222)")
            raise unwind.error
    class WaitTask(asyncio.Task):
        failed = False
        def fault(self, callback, at):
            if not self.failed and boundary == at and getattr(callback, "__name__", "") == "_on_completion":
                self.failed = True
                faulted.set()
                raise source.error
        def add_done_callback(self, callback, *args, **kwargs):
            self.fault(callback, "add")
            return super().add_done_callback(callback, *args, **kwargs)
        def remove_done_callback(self, callback):
            result = super().remove_done_callback(callback)
            self.fault(callback, "remove")
            return result
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **kwargs):
        if getattr(_actual_operation(coroutine), "cr_code", None) is hook.__code__:
            task = WaitTask(coroutine, loop=loop, eager_start=eager, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    actual_wait = asyncio.wait
    async def wait(*args, **kwargs):
        if asyncio.current_task() is owner:
            await entered.wait()
        return await actual_wait(*args, **kwargs)
    monkeypatch.setattr(asyncio, "wait", wait)
    owner = ownership.create_owned_task(manager._await_before_spawn_admission_deadline(
        hook, deadline=loop.time()+(0.05 if boundary == "remove" else 10), phase="original wait observer",
        agent=agent, on_expiry=lambda: fences.append(True)), retain_originals=True)
    try:
        loop.set_task_factory(factory)
        await asyncio.wait_for(faulted.wait(), 5)
        await asyncio.sleep(0)
        assert accepted and not owner.done(), "wait observation abandoned an accepted original hook"
        release.set()
        result = await ownership.await_owned_task(owner)
        source.assert_retained(result.error)
        unwind.assert_retained(result.error)
        assert all(task.done() for task in accepted)
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 2
        if boundary == "remove":
            assert fences == [True]
    finally:
        release.set()
        loop.set_task_factory(previous)
        for task in accepted:
            if not task.done():
                task.cancel()
        await ownership.await_owned_task(owner)
        for task in accepted:
            await ownership.await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_successful_registration_observation_failure_keeps_exact_rollback_owner(
    native_pg, onboarding_owner, monkeypatch, eager, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    accepted, calls = [], []
    attached = asyncio.Event()
    async def rollback():
        calls.append(rollback.scheduler_registration_nonce)
        await db.execute("INSERT INTO host_rollback_effects VALUES (224)")
    rollback.scheduler_registration_nonce = "original-exact-registration-nonce"
    async def register(*args):
        await db.execute("INSERT INTO host_rollback_effects VALUES (223)")
        return rollback
    manager._scheduler_polling_managed_by_host = True
    manager._scheduler_tenant_registration_hook = register
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    owner = ownership.create_owned_task(manager._begin_dynamic_scheduler_tenant_registration(
        "Original", agent.did, LocalAgentConfig(data_dir="original", port=8801)), retain_originals=True)
    try:
        loop.set_task_factory(observation_factory(lambda op: getattr(op, "cr_code", None) is register.__code__,
                                                 source.error, attached, accepted, previous, eager=eager))
        result = await ownership.await_owned_task(owner)
        source.assert_retained(result.error)
        assert attached.is_set() and len(accepted) == 1 and accepted[0].done()
        assert calls == [rollback.scheduler_registration_nonce], "physically prepared registration lost its original rollback"
        assert agent.did not in manager._scheduler_authority_by_did
        assert not manager.scheduler_lifecycle_lock(agent.did).locked()
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 2
    finally:
        loop.set_task_factory(previous)
        await ownership.await_owned_task(owner)
        for task in accepted:
            await ownership.await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("consumer", ["administrative", "terminal-drain", "quarantine"])
async def test_cold_offboarding_callback_failure_cannot_restore_authority_or_leak_record(
    native_pg, onboarding_owner, monkeypatch, tmp_path, installed, source_cancel, consumer,
):
    from kestrel_sovereign.features import isolated_runtime
    manager, agent, _ = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    pending, accepted = [], []
    entered, joining, waiting, release = asyncio.Event(), asyncio.Event(), asyncio.Event(), threading.Event()
    marker = tmp_path / "original-offboarding-thread-effect"
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def cleanup(*args):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10), "bounded fixture worker not released"
        marker.touch()
        return isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED
    monkeypatch.setattr(isolated_runtime, "remove_runtime_namespace", cleanup)
    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *args, **kwargs):
            if getattr(callback, "__name__", "") == "mark_complete":
                joining.set()
            if getattr(callback, "__name__", "") == "_on_completion":
                waiting.set()
            if getattr(callback, "__name__", "") == "retire_inflight":
                if installed:
                    super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
    def factory(loop, coroutine, **kwargs):
        if getattr(getattr(_actual_operation(coroutine), "cr_code", None), "co_name", "") == "cleanup_and_retire":
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    config = LocalAgentConfig(data_dir="cold-original", port=8801)
    manager._scheduler_revoked_names.add("Cold")
    manager._scheduler_revoked_dids.add(agent.did)
    owner = ownership.create_owned_task(manager._admit_agent_runtime_offboarding_identity(
        name="Cold", agent_id=agent.did, config=None, revoked=("Cold", config), pending_offboarding=pending), retain_originals=True)
    try:
        loop.set_task_factory(factory)
        result = await ownership.await_owned_task(owner)
        loop.set_task_factory(previous)
        await asyncio.wait_for(entered.wait(), 5)
        assert len(pending) == len(accepted) == 1, "accepted deletion was lost behind callback publication failure"
        assert pending[0].task is accepted[0] and not accepted[0].done()
        assert agent.did not in manager._scheduler_authority_by_did, "authority reopened over accepted destructive work"
        assert agent.did in manager._scheduler_revoked_dids
        if consumer == "quarantine":
            finisher = ownership.create_owned_task(manager._finish_agent_runtime_offboarding(
                pending[0], cancellation_already_observed=False), retain_originals=True)
            await asyncio.wait_for(waiting.wait(), 5)
            finisher.cancel("actual administrative offboarding observer withdrew")
            completed = await ownership.await_owned_task(finisher)
            assert completed.error is None and completed.result[0] is True
            failure = completed.result[1]
            source.assert_retained(failure)
            assert manager._quarantined_shutdown_reapers and not accepted[0].done()
            release.set()
            await ownership.await_owned_task(accepted[0])
            await asyncio.sleep(0)
        elif consumer == "terminal-drain":
            terminal = ownership.create_owned_task(manager.drain_quarantined_shutdowns(), retain_originals=True)
            await asyncio.wait_for(joining.wait(), 5)
            release.set()
            failure = (await ownership.await_owned_task(terminal)).error
        else:
            release.set()
            _, failure = await manager._finish_agent_runtime_offboarding(pending[0], cancellation_already_observed=False)
        source.assert_retained(failure)
        assert accepted[0].done() and marker.is_file()
        assert not manager._inflight_runtime_offboardings, "failed retirement callback leaked a completed original record"
        drained = await ownership.await_owned_task(ownership.create_owned_task(manager.drain_quarantined_shutdowns(), retain_originals=True))
        assert drained.error is None
        assert vars(manager).get("_quarantined_shutdown_control") is None
    finally:
        loop.set_task_factory(previous)
        release.set()
        await ownership.await_owned_task(owner)
        for task in accepted:
            await ownership.await_owned_task(task)
        manager._inflight_runtime_offboardings.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["public-drain", "descendant-fence", "child-setup", "fleet-removal", "fleet-drain", "fleet-spawn"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_later_manager_exit_keeps_native_prefix_and_whole_control_group(
    native_pg, onboarding_owner, monkeypatch, surface, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    ordinary = OriginalFailure.capture(OSError("original later boundary refused"))
    group = OriginalFailure.capture(BaseExceptionGroup("original whole lifecycle control", [source.error, UnexpectedControl("original sibling")]))
    tasks, calls = [], []
    async def spawn():
        await db.execute("INSERT INTO host_rollback_effects VALUES (225)")
        raise group.error if surface == "fleet-spawn" else source.error
    async def fail(*args, **kwargs):
        await db.execute("INSERT INTO host_rollback_effects VALUES (226)")
        raise ordinary.error
    if surface in ("public-drain", "descendant-fence"):
        task = ownership.create_owned_task(spawn(), retain_originals=True)
        tasks.append(task)
        manager._agent_operations["tracked"] = SimpleNamespace(kind="direct-spawn-test", spawn_task=task,
            canonical_name="tracked", spawn_parent=agent, committed=False)
        if surface == "public-drain":
            monkeypatch.setattr(manager, "_drain_quarantined_shutdowns_while_locked", fail)
            operation = manager.drain_quarantined_shutdowns()
        else:
            reads = []
            def witnesses():
                reads.append(True)
                if len(reads) > 1:
                    raise ordinary.error
                return ()
            monkeypatch.setattr(manager, "unsettled_spawn_authority_witnesses", witnesses)
            operation = manager.begin_terminal_descendant_spawn_fence(agent.did)
    elif surface == "child-setup":
        parent = "did:test:original-parent"
        async def relations():
            return {agent.did: (parent, "Child")}
        async def descendants(*args, **kwargs):
            await db.execute("INSERT INTO host_rollback_effects VALUES (225)")
            raise source.error
        monkeypatch.setattr(manager, "get_authoritative_spawn_relations", relations)
        monkeypatch.setattr(manager, "terminate_children", descendants)
        monkeypatch.setattr(manager, "_begin_destructive_spawn_retirement", fail)
        operation = manager.terminate_child(parent, "Child", offboard_runtime=True)
    else:
        manager._agents = {"First": agent, "Second": agent}
        if surface == "fleet-spawn":
            task = ownership.create_owned_task(spawn(), retain_originals=True)
            tasks.append(task)
            manager._agent_operations["tracked"] = SimpleNamespace(kind="direct-spawn-test", spawn_task=task,
                canonical_name="tracked", spawn_parent=agent, committed=False)
        async def remove(name, **kwargs):
            calls.append(name)
            await db.execute("INSERT INTO host_rollback_effects VALUES (?)", (227+len(calls),))
            manager._agents.pop(name)
            if surface == "fleet-removal" and name == "First":
                raise group.error
            return True
        async def drain(**kwargs):
            raise group.error
        monkeypatch.setattr(manager, "remove_agent", remove)
        if surface == "fleet-drain":
            monkeypatch.setattr(manager, "_drain_quarantined_shutdowns_while_locked", drain)
        operation = manager.shutdown_all()
    owner = ownership.create_owned_task(operation, retain_originals=True)
    try:
        result = await ownership.await_owned_task(owner)
        source.assert_retained(result.error)
        if surface.startswith("fleet"):
            group.assert_retained(result.error)
            assert calls == ["First", "Second"] and not manager._agents
        else:
            ordinary.assert_retained(result.error)
        assert not manager._quarantined_shutdown_drain_lock.locked()
        assert not manager._terminal_descendant_spawn_fence_members
    finally:
        await ownership.await_owned_task(owner)
        for task in tasks:
            await ownership.await_owned_task(task)
        manager._agent_operations.clear()
        manager._agents.clear()
