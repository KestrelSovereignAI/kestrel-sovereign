"""Exact native spawn/admission outcomes survive enclosing manager joins.

Native PG source and SQLite effects; tracking stubs do not prove signed spawn
authority or permit destructive offboarding, which these tests never request.
"""
import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_finite_error, _explicit_control_chain
from kestrel_sovereign.spawn.mandate import PersistedSpawnMandateExpiredError
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["caller", "expiry"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_spawn_deadline_unwind_keeps_native_original(
    native_pg, onboarding_owner, monkeypatch, ending, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    entered, fenced, withdrawals = asyncio.Event(), [], []
    owner, actual_wait = None, asyncio.wait
    async def observe(*args, **kwargs):
        try:
            return await actual_wait(*args, **kwargs)
        except asyncio.CancelledError as error:
            if asyncio.current_task() is owner:
                withdrawals.append(error)
            raise
    monkeypatch.setattr(asyncio, "wait", observe)
    async def hook():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await db.execute("INSERT INTO host_rollback_effects VALUES (120)")
            raise source.error
    deadline = asyncio.get_running_loop().time() + (0.05 if ending == "expiry" else 30)
    owner = create_owned_task(manager._await_before_spawn_admission_deadline(
        hook, deadline=deadline, phase="declared native component hook", agent=agent,
        on_expiry=lambda: fenced.append(True),
    ), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if ending == "caller":
            owner.cancel("actual original admission observer withdrew")
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        if ending == "caller":
            assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        else:
            assert fenced == [True]
            assert any(isinstance(error, PersistedSpawnMandateExpiredError)
                       for error in _explicit_control_chain(result.error))
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
        assert not manager._expired_admission_operations
    finally:
        await await_owned_task(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["fleet", "removal", "descendants"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_spawn_join_retains_native_source_instead_of_treating_it_as_ordinary_failure(
    native_pg, onboarding_owner, monkeypatch, surface, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    async def spawn():
        await db.execute("INSERT INTO host_rollback_effects VALUES (121)")
        raise source.error
    task = create_owned_task(spawn(), retain_originals=True)
    admission = SimpleNamespace(kind="direct-spawn-test", spawn_task=task, canonical_name="tracked",
                                spawn_parent=agent, committed=False)
    manager._agent_operations["tracked"] = admission
    if surface == "fleet":
        op = manager._join_admitted_spawn_operations()
    elif surface == "removal":
        op = manager._join_active_spawn_before_removal("tracked")
    else:
        op = manager.begin_terminal_descendant_spawn_fence(agent.did)
    owner = create_owned_task(op, retain_originals=True)
    try:
        result = await await_owned_task(owner)
        terminal = (execution_finite_error(*result.result[1])
                    if surface == "fleet" and result.error is None else result.error)
        source.assert_retained(terminal)
        assert task.done() and await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    finally:
        result = await await_owned_task(owner)
        await await_owned_task(task)
        if surface == "descendants" and result.error is None:
            manager.end_terminal_descendant_spawn_fence(result.result)
        manager._agent_operations.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("removal_failed", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_child_reconciliation_wrapper_keeps_canonical_native_cause(
    native_pg, onboarding_owner, monkeypatch, removal_failed, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    ordinary = OriginalFailure.capture(OSError("actual declared child removal failed"))
    parent = "did:test:component-parent"
    async def relations():
        return {agent.did: (parent, "Child")}
    async def remove(name, **kwargs):
        assert kwargs["offboard_runtime"] is False
        await db.execute("INSERT INTO host_rollback_effects VALUES (122)")
        if removal_failed:
            raise ordinary.error
        return True
    async def reconcile(*args):
        await db.execute("INSERT INTO host_rollback_effects VALUES (123)")
        raise source.error
    monkeypatch.setattr(manager, "get_authoritative_spawn_relations", relations)
    monkeypatch.setattr(manager, "remove_agent", remove)
    monkeypatch.setattr(manager, "_prune_child_tracking_if_fully_removed", reconcile)
    result = await await_owned_task(create_owned_task(manager.terminate_child(parent, "Child"), retain_originals=True))
    source.assert_retained(result.error)
    if removal_failed:
        ordinary.assert_retained(result.error)
    assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 2

@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_failed_expiry_fence_still_joins_original_native_unwind(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    fence = OriginalFailure.capture(OSError("original expiry fence reporting failed"))
    entered, fenced, release, children = asyncio.Event(), asyncio.Event(), asyncio.Event(), []
    async def hook():
        children.append(asyncio.current_task())
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
            await db.execute("INSERT INTO host_rollback_effects VALUES (124)")
            raise source.error
    def fail_fence():
        fenced.set()
        raise fence.error
    owner = create_owned_task(manager._await_before_spawn_admission_deadline(
        hook, deadline=asyncio.get_running_loop().time()+0.05,
        phase="declared native unwind with failed fence", agent=agent, on_expiry=fail_fence,
    ), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(fenced.wait(), 5)
        assert children and not children[0].done()
        assert not owner.done(), "failed expiry callback abandoned its original native hook"
        release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        fence.assert_retained(result.error)
        assert any(isinstance(error, PersistedSpawnMandateExpiredError) for error in _explicit_control_chain(result.error))
        assert all(child.done() for child in children)
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    finally:
        release.set()
        await await_owned_task(owner)
        for child in children:
            await await_owned_task(child)

@pytest.mark.asyncio
@pytest.mark.parametrize("callback", ["retire_delegate", "retire"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_failed_admission_bookkeeping_publication_cannot_abandon_original_hook(
    native_pg, onboarding_owner, monkeypatch, callback, source_cancel,
):
    from tests.unit.test_host_context_native_cleanup import _actual_operation
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    entered, unwinding, release, accepted, hits = asyncio.Event(), asyncio.Event(), asyncio.Event(), [], []
    async def hook():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            unwinding.set()
            await release.wait()
            await db.execute("INSERT INTO host_rollback_effects VALUES (125)")
            return "physically joined original"
    class PublicationTask(asyncio.Task):
        failed = False
        def add_done_callback(self, observer, *args, **kwargs):
            if not self.failed and getattr(observer, "__name__", "") == callback:
                self.failed = True
                hits.append(self)
                raise source.error
            return super().add_done_callback(observer, *args, **kwargs)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **kwargs):
        if getattr(_actual_operation(coroutine), "cr_code", None) is hook.__code__:
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    loop.set_task_factory(factory)
    owner = create_owned_task(manager._await_before_spawn_admission_deadline(
        hook, deadline=asyncio.get_running_loop().time()+0.05,
        phase="declared native hook bookkeeping", agent=agent,
        admission=SimpleNamespace() if callback == "retire_delegate" else None,
    ), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert hits or callback == "retire"
        if callback == "retire_delegate" and owner.done():
            pytest.fail("delegate observer publication abandoned an accepted original hook")
        await asyncio.wait_for(unwinding.wait(), 5)
        assert hits == accepted and accepted and not accepted[0].done()
        assert not owner.done(), "expiry observer publication abandoned an accepted original hook"
        release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert all(task.done() for task in accepted)
        assert not manager._expired_admission_operations
        assert not manager._admission_delegate_tasks
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    finally:
        release.set()
        loop.set_task_factory(previous)
        for task in accepted:
            if not task.done():
                task.cancel()
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)

@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
async def test_publication_and_unexpected_observation_keep_whole_native_source_group(
    native_pg, onboarding_owner, monkeypatch, eager,
):
    from tests.unit.test_host_context_native_cleanup import _actual_operation
    manager, agent, db = onboarding_owner
    native = await _original_control(native_pg[0], monkeypatch, False)
    opening = OriginalFailure.capture(BaseExceptionGroup("genuine original publication group",
                                                       [native, OSError("original publication sibling")]))
    class UnexpectedObservation(BaseException):
        pass
    unexpected = UnexpectedObservation("actual original join observation failed")
    unexpected.__cause__ = native
    observed = OriginalFailure.capture(unexpected)
    accepted, hits, owner = [], [], None
    async def hook():
        await db.execute("INSERT INTO host_rollback_effects VALUES (126)")
    class PublicationTask(asyncio.Task):
        opened = False
        observed = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not self.opened and getattr(callback, "__name__", "") == "close_unstarted":
                self.opened = True
                hits.append("publication")
                raise opening.error
            return super().add_done_callback(callback, *args, **kwargs)
        def remove_done_callback(self, callback):
            result = super().remove_done_callback(callback)
            if not self.observed and self.done() and getattr(callback, "__name__", "") == "mark_complete":
                self.observed = True
                hits.append("observation")
                raise unexpected
            return result
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **kwargs):
        if getattr(_actual_operation(coroutine), "cr_code", None) is hook.__code__:
            task = PublicationTask(coroutine, loop=loop, eager_start=eager, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    loop.set_task_factory(factory)
    owner = create_owned_task(manager._await_before_spawn_admission_deadline(
        hook, deadline=asyncio.get_running_loop().time()+10,
        phase="declared source-group coordinator", agent=agent,
    ), retain_originals=True)
    try:
        result = await await_owned_task(owner)
        opening.assert_retained(result.error)
        observed.assert_retained(result.error)
        assert hits == ["publication", "observation"] and len(accepted) == 1
        assert accepted[0].done()
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 0
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
