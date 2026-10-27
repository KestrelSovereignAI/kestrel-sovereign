"""Reader/drain/fence terminal exits keep original control and observations.

Local PostgreSQL source receipts, SQLite effects and original Task/lock joins.
Declared admission records are not signed authority or whole-Agent readiness.
"""
import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign.multi_agent import agent_manager as manager_module
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.integration.test_native_completion_vs_observation import observation_factory
from tests.shared.original_task_admission import assert_unknown_task_admission
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


class UnexpectedControl(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["drain-admission", "spawn-acquire", "spawn-wait"])
@pytest.mark.parametrize("grouped", [False, True])
async def test_manager_admission_diagnostic_keeps_original_unexpected_control(
    onboarding_owner, surface, grouped,
):
    from tests.utils.native_admission_fault import NativeAdmissionFault
    manager, agent, _ = onboarding_owner
    leaf = UnexpectedControl("actual original admission refusal")
    source = OriginalFailure.capture(leaf)
    publication = BaseExceptionGroup("original admission group", [leaf]) if grouped else leaf
    publication_source = OriginalFailure.capture(publication)
    admission = manager_module.AgentOperationAdmission(
        name="Child", canonical_name="child", kind="spawn",
        registration_epoch=manager._agent_registration_shutdown_epoch,
        owner_task=asyncio.current_task(),
    )
    if surface == "spawn-wait":
        manager._quarantined_shutdown_handoffs_sealed = True
        manager._quarantined_shutdown_handoffs_open.clear()
    def predicate(operation):
        code = getattr(operation, "cr_code", None)
        if surface == "spawn-wait":
            frame = getattr(operation, "cr_frame", None)
            return (code is asyncio.Event.wait.__code__ and frame is not None
                    and frame.f_locals.get("self") is manager._quarantined_shutdown_handoffs_open)
        return (code is asyncio.Lock.acquire.__code__ if surface == "drain-admission"
                else getattr(code, "co_name", "") == "acquire_manager_lock")
    fault = NativeAdmissionFault(predicate, publication, "refuse")
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    attached, accepted = asyncio.Event(), []
    operation = (manager.drain_quarantined_shutdowns() if surface == "drain-admission"
                 else manager._handoff_failed_spawn_cleanup(admission, agent))
    owner = ownership.create_owned_task(operation, retain_originals=True)
    try:
        if surface == "drain-admission":
            with fault:
                result = await ownership.await_owned_task(owner)
            fault.assert_terminal_unstarted()
        else:
            # These are failures of the accepted phase's completion observer,
            # not refused pre-entry retirement recipes (which carry native
            # debt and have a different, already-protected contract).
            loop.set_task_factory(observation_factory(
                predicate, publication, attached, accepted, previous, eager=False))
            if surface == "spawn-wait":
                await asyncio.wait_for(attached.wait(), 5)
                manager._quarantined_shutdown_handoffs_open.set()
            result = await ownership.await_owned_task(owner)
            assert attached.is_set() and len(accepted) == 1 and accepted[0].done()
        source.assert_retained(result.error)
        if surface == "drain-admission":
            assert_unknown_task_admission(result.error, publication, source=publication_source)
        else:
            assert not isinstance(result.error, Exception), "admission control became ordinary availability"
        assert not manager._lock.locked() and not manager._quarantined_shutdown_drain_lock.locked()
        assert not manager._quarantined_shutdown_reapers
    finally:
        loop.set_task_factory(previous)
        manager._quarantined_shutdown_handoffs_sealed = False
        manager._quarantined_shutdown_handoffs_open.set()
        await ownership.await_owned_task(owner)
        for task in accepted:
            await ownership.await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["ordinary-owner", "ordinary-yield"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_reader_every_exit_includes_already_harvested_native_summary(
    native_pg, onboarding_owner, monkeypatch, exit_kind, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    later = OriginalFailure.capture(OSError("actual later ordinary completion failed"))
    entered, release, attached = asyncio.Event(), asyncio.Event(), asyncio.Event()
    owner, accepted = None, []
    async def native():
        await db.execute("INSERT INTO host_rollback_effects VALUES (381)")
        raise source.error
    first = ownership.create_owned_task(native(), retain_originals=True)
    manager._retain_quarantined_cleanup(name="Child", agent_id=agent.did, task=first)
    await ownership.await_owned_task(first)
    await asyncio.sleep(0)
    assert not manager._quarantined_shutdown_reapers
    source.assert_retained(manager._quarantined_shutdown_control)
    async def ordinary():
        await release.wait()
        await db.execute("INSERT INTO host_rollback_effects VALUES (382)")
        if exit_kind == "ordinary-owner":
            raise later.error
    second = ownership.create_owned_task(ordinary(), retain_originals=True)
    manager._retain_quarantined_cleanup(name="Child", agent_id=agent.did, task=second)
    actual_join = manager_module.await_lifecycle_task_completion
    async def join(task, **kwargs):
        if task is second:
            entered.set()
        return await actual_join(task, **kwargs)
    monkeypatch.setattr(manager_module, "await_lifecycle_task_completion", join)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def predicate(operation):
        return (asyncio.current_task() is owner and
                getattr(operation, "cr_code", None) is asyncio.sleep.__code__)
    if exit_kind == "ordinary-yield":
        loop.set_task_factory(observation_factory(predicate, later.error, attached, accepted, previous, eager=False))
    owner = ownership.create_owned_task(manager.await_child_termination_reconciliation(
        "did:test:parent", "Child"), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        release.set()
        result = await ownership.await_owned_task(owner)
        source.assert_retained(result.error)
        later.assert_retained(result.error)
        assert second.done() and not manager._quarantined_shutdown_reapers
        if exit_kind == "ordinary-yield":
            assert attached.is_set() and len(accepted) == 1 and accepted[0].done()
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 2
    finally:
        loop.set_task_factory(previous)
        release.set()
        await ownership.await_owned_task(first)
        await ownership.await_owned_task(second)
        await ownership.await_owned_task(owner)
        for task in accepted:
            await ownership.await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["drain-yield", "drain-reconciliation", "reader-reconciliation"])
@pytest.mark.parametrize("grouped", [False, True])
async def test_boundary_completion_observer_cannot_demote_unexpected_control(
    onboarding_owner, monkeypatch, surface, grouped,
):
    manager, agent, db = onboarding_owner
    leaf = UnexpectedControl("actual original completion observer control")
    failure = OriginalFailure.capture(leaf)
    publication = BaseExceptionGroup("whole unexpected observer", [leaf]) if grouped else leaf
    attached, release, observed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    accepted, originals, owner = [], [], None
    if surface == "drain-yield":
        async def cleanup():
            await release.wait()
            await db.execute("INSERT INTO host_rollback_effects VALUES (391)")
        original = ownership.create_owned_task(cleanup(), retain_originals=True)
        originals.append(original)
        manager._retain_quarantined_cleanup(name="Child", agent_id=agent.did, task=original)
        actual_join = manager_module.await_lifecycle_task_completion
        async def join(task, **kwargs):
            if task is original:
                observed.set()
            return await actual_join(task, **kwargs)
        monkeypatch.setattr(manager_module, "await_lifecycle_task_completion", join)
    else:
        # Keep actual pruning/manager-lock work; only observe entry to that
        # real Task so its completion callback is attached before it finishes.
        await manager._lock.acquire()
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def predicate(operation):
        code = getattr(operation, "cr_code", None)
        if surface == "drain-yield":
            return asyncio.current_task() is owner and code is asyncio.sleep.__code__
        expected = (manager._prune_all_fully_removed_child_tracking.__func__.__code__
                    if surface == "drain-reconciliation" else
                    manager._prune_child_tracking_if_fully_removed.__func__.__code__)
        return code is expected
    loop.set_task_factory(observation_factory(predicate, publication, attached, accepted, previous, eager=False))
    owner = ownership.create_owned_task(
        manager.await_child_termination_reconciliation("did:test:parent", "Child")
        if surface == "reader-reconciliation" else manager.drain_quarantined_shutdowns(),
        retain_originals=True)
    try:
        if surface == "drain-yield":
            await asyncio.wait_for(observed.wait(), 5)
            release.set()
        else:
            # The drain first needs the same lock for its handoff seal.
            # Releasing now still leaves pruning as a deferred original Task.
            manager._lock.release()
        result = await ownership.await_owned_task(owner)
        failure.assert_retained(result.error)
        assert not isinstance(result.error, Exception), "unexpected control became ordinary availability"
        assert attached.is_set() and len(accepted) == 1 and accepted[0].done()
        assert all(task.done() for task in originals)
        if originals:
            assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
        assert not manager._quarantined_shutdown_handoffs_sealed
    finally:
        loop.set_task_factory(previous)
        release.set()
        if manager._lock.locked():
            manager._lock.release()
        for task in originals:
            await ownership.await_owned_task(task)
        await ownership.await_owned_task(owner)
        for task in accepted:
            await ownership.await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("release_failure", [False, True])
async def test_descendant_fence_keeps_every_actual_withdrawal_and_release_error(
    onboarding_owner, monkeypatch, release_failure,
):
    manager, agent, db = onboarding_owner
    releases = [asyncio.Event(), asyncio.Event()]
    entered = [asyncio.Event(), asyncio.Event()]
    observations, originals, owner = [], [], None
    later = OriginalFailure.capture(OSError("actual original fence release failed"))
    async def spawn(index):
        await releases[index].wait()
        await db.execute("INSERT INTO host_rollback_effects VALUES (?)", (401+index,))
    for index in range(2):
        task = ownership.create_owned_task(spawn(index), retain_originals=True)
        originals.append(task)
        manager._agent_operations[str(index)] = SimpleNamespace(
            kind="direct-spawn-test", spawn_task=task, canonical_name=str(index),
            spawn_parent=agent, committed=False)
    actual_owned = ownership.await_owned_task
    async def observe(task, *args, **kwargs):
        if asyncio.current_task() is owner and task in originals:
            entered[originals.index(task)].set()
        outcome = await actual_owned(task, *args, **kwargs)
        if asyncio.current_task() is owner and task in originals and outcome.cancellation is not None:
            observations.append(OriginalFailure.capture(outcome.cancellation))
        return outcome
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    actual_end = manager.end_terminal_descendant_spawn_fence
    def end(token):
        actual_end(token)
        if release_failure:
            raise later.error
    monkeypatch.setattr(manager, "end_terminal_descendant_spawn_fence", end)
    owner = ownership.create_owned_task(manager.begin_terminal_descendant_spawn_fence(agent.did), retain_originals=True)
    try:
        for index in range(2):
            await asyncio.wait_for(entered[index].wait(), 5)
            owner.cancel(f"actual fence observer withdrawal {index}")
            await asyncio.sleep(0)
            releases[index].set()
        result = await actual_owned(owner)
        assert len(observations) == 2 and observations[0].error is not observations[1].error
        for observation in observations:
            observation.assert_retained(result.error)
        if release_failure:
            later.assert_retained(result.error)
        assert all(task.done() for task in originals)
        assert not manager._terminal_descendant_spawn_fence_members
        assert not manager._terminal_descendant_spawn_fences
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 2
    finally:
        for release in releases:
            release.set()
        for task in originals:
            await actual_owned(task)
        manager._agent_operations.clear()
        await actual_owned(owner)


@pytest.mark.asyncio
async def test_descendant_fence_handled_count_does_not_invent_new_observation(onboarding_owner):
    manager, agent, db = onboarding_owner
    entered = asyncio.Event()
    handled = []
    async def spawn():
        await entered.wait()
        await db.execute("INSERT INTO host_rollback_effects VALUES (411)")
    original = ownership.create_owned_task(spawn(), retain_originals=True)
    manager._agent_operations["declared"] = SimpleNamespace(
        kind="direct-spawn-test", spawn_task=original, canonical_name="declared",
        spawn_parent=agent, committed=False)
    async def operation():
        current = asyncio.current_task()
        current.cancel("previously delivered and explicitly handled")
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as error:
            handled.append(OriginalFailure.capture(error))
        assert current.cancelling() == 1
        entered.set()
        return await manager.begin_terminal_descendant_spawn_fence(agent.did)
    owner = ownership.create_owned_task(operation(), retain_originals=True)
    try:
        result = await ownership.await_owned_task(owner)
        assert len(handled) == 1
        assert result.error is None, "a handled cancellation count manufactured a new fence cancellation"
        assert result.result.nonce in manager._terminal_descendant_spawn_fence_members
        manager.end_terminal_descendant_spawn_fence(result.result)
        assert original.done() and await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    finally:
        entered.set()
        await ownership.await_owned_task(original)
        manager._agent_operations.clear()
        await ownership.await_owned_task(owner)
