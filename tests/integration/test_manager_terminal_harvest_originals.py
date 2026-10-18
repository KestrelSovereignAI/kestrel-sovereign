"""Terminal consumers harvest original physical owners without callback ACKs.

Real local PostgreSQL source receipts and SQLite effects; declared component
protocols do not certify whole-Agent readiness, signed authority or funding.
"""
import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign.multi_agent import agent_manager as manager_module
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


class UnexpectedControl(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_original_join_keeps_cyclic_native_component_beside_independent_observer(
    native_pg, onboarding_owner, monkeypatch, eager, source_cancel,
):
    _, _, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    first = RuntimeError("original cyclic physical carrier")
    second = BaseExceptionGroup("original cyclic publication carrier", [source.error])
    first.__cause__ = second
    second.__cause__ = first
    physical, publication, independent = (OriginalFailure.capture(error) for error in
        (first, second, OSError("independent original completion observer")))
    hits, accepted = [], []
    async def operation():
        await db.execute("INSERT INTO host_rollback_effects VALUES (301)")
        raise physical.error
    class ObservationTask(asyncio.Task):
        def add_done_callback(self, callback, *args, **kwargs):
            if getattr(callback, "__name__", "") == "mark_complete":
                hits.append("publish")
                raise publication.error
            return super().add_done_callback(callback, *args, **kwargs)
        def remove_done_callback(self, callback):
            result = super().remove_done_callback(callback)
            if getattr(callback, "__name__", "") == "mark_complete":
                hits.append("retire")
                raise independent.error
            return result
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **kwargs):
        if getattr(_actual_operation(coroutine), "cr_code", None) is operation.__code__:
            task = ObservationTask(coroutine, loop=loop, eager_start=eager, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    try:
        loop.set_task_factory(factory)
        original = ownership.create_owned_task(operation(), retain_originals=True)
        loop.set_task_factory(previous)
        result = await ownership.await_owned_task(original)
        for error in (source, physical, publication, independent):
            error.assert_retained(result.error)
        assert hits == ["publish", "retire"] and accepted == [original] and original.done()
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=301") == 1
    finally:
        loop.set_task_factory(previous)
        for task in accepted:
            await ownership.await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["quarantine-drain", "quarantine-child", "budget-drain", "budget-direct"])
@pytest.mark.parametrize("mode", ["before", "after"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_terminal_join_retires_completed_record_after_callback_publication_failure(
    native_pg, onboarding_owner, monkeypatch, surface, mode, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    release, joined, observing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    accepted, hits = [], []
    worker = owner = record = None
    callback_name = ("observe_reaper_completion" if surface.startswith("quarantine")
                     else "observe_budget_release_completion")
    actual_join = manager_module.await_lifecycle_task_completion
    async def cleanup():
        await release.wait()
        await db.execute("INSERT INTO host_rollback_effects VALUES (241)")
    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *args, **kwargs):
            if getattr(callback, "__name__", "") == callback_name and not hits:
                hits.append(self)
                if mode == "after":
                    super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **kwargs):
        if getattr(_actual_operation(coroutine), "cr_code", None) is cleanup.__code__:
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    async def observed(task, **kwargs):
        if task is worker:
            observing.set()
        try:
            return await actual_join(task, **kwargs)
        finally:
            if task is worker and task.done():
                joined.set()
    async def consume():
        if surface == "quarantine-child":
            return await manager.await_child_termination_reconciliation("did:test:declared-parent", "Harvest")
        if surface == "budget-direct":
            try:
                observing.set()
                return await manager._await_child_budget_release_cancellation_safe(worker)
            finally:
                joined.set()
        return await manager.drain_quarantined_shutdowns()
    try:
        loop.set_task_factory(factory)
        if surface.startswith("quarantine"):
            worker = ownership.create_owned_task(cleanup(), retain_originals=True)
            try:
                manager._retain_quarantined_cleanup(name="Harvest", agent_id=agent.did, task=worker)
            except BaseException as opening:
                source.assert_retained(opening)
            record = next(iter(manager._quarantined_shutdown_reapers.values()))
        else:
            record = manager._start_child_budget_release("Harvest", release_operation=cleanup)
            worker = record.task
        loop.set_task_factory(previous)
        assert hits == accepted == [worker] and not worker.done()
        monkeypatch.setattr(manager_module, "await_lifecycle_task_completion", observed)
        owner = ownership.create_owned_task(consume(), retain_originals=True)
        await asyncio.wait_for(observing.wait(), 5)
        release.set()
        await asyncio.wait_for(joined.wait(), 5)
        await asyncio.sleep(0)
        assert not any(item.task is worker for item in manager._quarantined_shutdown_reapers.values()), "completed quarantine owner still depends on missing callback"
        assert not any(item.task is worker for item in manager._inflight_removal_budget_releases.values()), "completed budget owner still depends on missing callback"
        assert "Harvest" not in manager._inflight_removal_budget_releases_by_child
        outcome = await ownership.await_owned_task(owner)
        source.assert_retained(outcome.error)
        assert worker.done() and await db.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=241") == 1
        counts = (len(manager._unsafe_quarantined_shutdown_failures), len(manager._unsafe_removal_budget_release_failures))
        await ownership.await_owned_task(ownership.create_owned_task(manager.drain_quarantined_shutdowns(), retain_originals=True))
        assert counts == (len(manager._unsafe_quarantined_shutdown_failures), len(manager._unsafe_removal_budget_release_failures)), "repeat terminal observation duplicated bounded history"
    finally:
        loop.set_task_factory(previous)
        release.set()
        if worker is not None:
            await ownership.await_owned_task(worker)
            # Fixture-only release lets a deliberately broken negative finish;
            # assertions above, never this teardown, prove production retirement.
            for key, item in tuple(manager._quarantined_shutdown_reapers.items()):
                if item.task is worker:
                    manager._quarantined_shutdown_reapers.pop(key)
            for key, item in tuple(manager._inflight_removal_budget_releases.items()):
                if item.task is worker:
                    manager._inflight_removal_budget_releases.pop(key)
                    manager._inflight_removal_budget_releases_by_child.pop(item.child_name, None)
        if owner is not None:
            await ownership.await_owned_task(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["children", "topology", "pruning", "removal", "descendants"])
@pytest.mark.parametrize("through_cause", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_whole_native_group_survives_sweep_filters_and_independent_attempts(
    native_pg, onboarding_owner, monkeypatch, surface, through_cause, source_cancel,
):
    from kestrel_sovereign.spawn.mandate import SpawnMandate
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    error = BaseExceptionGroup("whole original source group", [UnexpectedControl("original nonordinary leaf")]
                              if through_cause else [source.error, UnexpectedControl("original nonordinary leaf")])
    if through_cause:
        error.__cause__ = source.error
    group = OriginalFailure.capture(error)
    prior = OriginalFailure.capture(OSError("original independent setup failure"))
    calls = []
    async def children(*args):
        return ["First", "Second"]
    async def terminate(parent, name, **kwargs):
        calls.append(name)
        await db.execute("INSERT INTO host_rollback_effects VALUES (?)", (250+len(calls),))
        if name == "First":
            raise group.error
        return True
    async def topology():
        raise group.error if surface == "topology" else AssertionError("unexpected topology probe")
    async def remove(name, **kwargs):
        calls.append(name)
        await db.execute("INSERT INTO host_rollback_effects VALUES (?)", (250+len(calls),))
        manager._agents.pop(name)
        if surface == "removal" and name == "First":
            raise group.error
        return True
    async def prune(*args):
        await db.execute("INSERT INTO host_rollback_effects VALUES (254)")
        raise group.error
    if surface == "children":
        monkeypatch.setattr(manager, "get_authoritative_children", children)
        monkeypatch.setattr(manager, "terminate_child", terminate)
        operation = manager.terminate_children("did:test:declared-parent")
    elif surface == "descendants":
        async def relations():
            return {agent.did: ("did:test:declared-parent", "Child")}
        async def descendants(*args, **kwargs):
            raise group.error
        async def setup(*args, **kwargs):
            calls.append("setup")
            await db.execute("INSERT INTO host_rollback_effects VALUES (251)")
            raise prior.error
        monkeypatch.setattr(manager, "get_authoritative_spawn_relations", relations)
        monkeypatch.setattr(manager, "terminate_children", descendants)
        monkeypatch.setattr(manager, "_begin_destructive_spawn_retirement", setup)
        operation = manager.terminate_child("did:test:declared-parent", "Child", offboard_runtime=True)
    else:
        manager._agents = {"First": agent, "Second": agent}
        monkeypatch.setattr(manager, "remove_agent", remove)
        if surface == "topology":
            monkeypatch.setattr(manager, "_verified_spawn_relations", topology)
        if surface == "pruning":
            # Tracking component only, deliberately not signed authority.
            manager._child_mandates["First"] = SpawnMandate(parent_did=agent.did, child_did=agent.did, authority_committed=False)
            monkeypatch.setattr(manager, "_prune_child_tracking_if_fully_removed", prune)
        operation = manager.shutdown_all()
    owner = ownership.create_owned_task(operation, retain_originals=True)
    try:
        result = await ownership.await_owned_task(owner)
        source.assert_retained(result.error)
        group.assert_retained(result.error)
        assert calls == (["setup"] if surface == "descendants" else ["First", "Second"])
        if surface == "descendants":
            prior.assert_retained(result.error)
        assert not manager._quarantined_shutdown_drain_lock.locked()
        assert not manager._terminal_descendant_spawn_fence_members
    finally:
        await ownership.await_owned_task(owner)
        manager._agents.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["setup", "terminal"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_child_settlement_raised_group_still_attempts_compensation_or_reconciliation(
    native_pg, onboarding_owner, monkeypatch, surface, source_cancel,
):
    from kestrel_sovereign.spawn.mandate import SpawnMandate
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    group = OriginalFailure.capture(BaseExceptionGroup("original settlement source", [source.error, UnexpectedControl("original sibling")]))
    opening = OriginalFailure.capture(OSError("original setup or removal failed"))
    parent, calls = "did:test:declared-parent", []
    async def relations():
        return {agent.did: (parent, "Child")}
    async def settlement(*args):
        calls.append("settle")
        await db.execute("INSERT INTO host_rollback_effects VALUES (271)")
        raise group.error
    async def rollback():
        calls.append("rollback")
        await db.execute("INSERT INTO host_rollback_effects VALUES (272)")
    async def remove_hook(*args):
        return rollback
    async def begin(*args, **kwargs):
        return None
    def withdraw(*args, **kwargs):
        raise opening.error
    async def remove(*args, **kwargs):
        raise opening.error
    async def reconcile(*args):
        calls.append("reconcile")
        await db.execute("INSERT INTO host_rollback_effects VALUES (272)")
        return False
    monkeypatch.setattr(manager, "get_authoritative_spawn_relations", relations)
    monkeypatch.setattr(manager, "_settle_destructive_spawn_retirement", settlement)
    monkeypatch.setattr(manager, "_prune_child_tracking_if_fully_removed", reconcile)
    if surface == "setup":
        manager._child_mandates["Child"] = SpawnMandate(parent_did=parent, child_did=agent.did, ttl_seconds=0, authority_committed=False)
        manager._created_agent_registration_removal_hook = remove_hook
        monkeypatch.setattr(manager, "_begin_destructive_spawn_retirement", begin)
        monkeypatch.setattr(manager, "_withdraw_committed_spawn_startup_registration", withdraw)
    else:
        monkeypatch.setattr(manager, "remove_agent", remove)
    owner = ownership.create_owned_task(manager.terminate_child(parent, "Child", offboard_runtime=surface == "setup"), retain_originals=True)
    try:
        result = await ownership.await_owned_task(owner)
        source.assert_retained(result.error)
        group.assert_retained(result.error)
        opening.assert_retained(result.error)
        assert calls == ["settle", "rollback" if surface == "setup" else "reconcile"], "raised settlement join abandoned independent bookkeeping cleanup"
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 2
        assert not manager._persistent_spawn_offboarding
    finally:
        await ownership.await_owned_task(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_child_specific_reader_preserves_native_result_and_actual_withdrawal(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    owner = None
    async def cleanup():
        await release.wait()
        await db.execute("INSERT INTO host_rollback_effects VALUES (281)")
        raise source.error
    original = ownership.create_owned_task(cleanup(), retain_originals=True)
    manager._retain_quarantined_cleanup(name="Child", agent_id=agent.did, task=original)
    actual_join = ownership.await_owned_task
    async def observed(task, *args, **kwargs):
        if task is original and asyncio.current_task() is owner:
            entered.set()
        result = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is owner and result.cancellation is not None:
            withdrawals.append(OriginalFailure.capture(result.cancellation))
        return result
    monkeypatch.setattr(ownership, "await_owned_task", observed)
    owner = ownership.create_owned_task(manager.await_child_termination_reconciliation("did:test:declared-parent", "Child"), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual child-specific completion observer withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await actual_join(owner)
        source.assert_retained(result.error)
        assert len(withdrawals) == 1
        withdrawals[0].assert_retained(result.error)
        assert original.done() and not manager._quarantined_shutdown_reapers
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    finally:
        release.set()
        await actual_join(original)
        await actual_join(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["instance", "class"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_ttl_consumer_preserves_prior_terminal_and_late_native_reader_failure(
    native_pg, onboarding_owner, monkeypatch, entry, source_cancel,
):
    from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle, SpawnMode, SpawnStatus, _TrackedChild
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    prior = OriginalFailure.capture(manager_module.RuntimeOffboardingRetainedError(
        agent_name="Child", agent_id=agent.did, runtime_path=None,
        cause=OSError("original prior offboarding failure"), cleanup_pending=True))
    entered, release = asyncio.Event(), asyncio.Event()
    async def cleanup():
        await release.wait()
        await db.execute("INSERT INTO host_rollback_effects VALUES (291)")
        raise source.error
    original = ownership.create_owned_task(cleanup(), retain_originals=True)
    manager._retain_quarantined_cleanup(name="Child", agent_id=agent.did, task=original)
    # Declared outer lifecycle protocol. The reader and original cleanup Task
    # are real; no signed removal or whole-Agent readiness is claimed.
    class Platform:
        def get_agent(self, name):
            return None
        async def terminate_child(self, parent, child, **kwargs):
            raise prior.error
    platform = Platform()
    async def reconcile(parent, child):
        entered.set()
        return await manager.await_child_termination_reconciliation(parent, child)
    if entry == "instance":
        platform.await_child_termination_reconciliation = reconcile
    else:
        async def class_reconcile(self, parent, child):
            return await reconcile(parent, child)
        Platform.await_child_termination_reconciliation = class_reconcile
    lifecycle = SpawnedAgentLifecycle(platform)
    tracked = _TrackedChild("Child", agent.did, "did:test:declared-parent", SpawnMode.PERSISTENT, 0, "declared TTL consumer")
    lifecycle._tracked["Child"] = tracked
    owner = ownership.create_owned_task(lifecycle._terminate_and_cleanup(
        "Child", SpawnStatus.TIMED_OUT, _terminal_descendant_fence_active=True), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        release.set()
        result = await ownership.await_owned_task(owner)
        source.assert_retained(result.error)
        prior.assert_retained(result.error)
        assert lifecycle._tracked.get("Child") is tracked and not lifecycle._results
        assert original.done() and await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    finally:
        release.set()
        await ownership.await_owned_task(original)
        await ownership.await_owned_task(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["instance", "class"])
@pytest.mark.parametrize("reconciled", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_ttl_successful_reader_never_reconstructs_actual_native_withdrawal(
    native_pg, onboarding_owner, monkeypatch, entry, reconciled, source_cancel,
):
    from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle, SpawnMode, SpawnStatus, _TrackedChild
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    prior = OriginalFailure.capture(manager_module.RuntimeOffboardingRetainedError(
        agent_name="Child", agent_id=agent.did, runtime_path=None, cause=source.error, cleanup_pending=True))
    entered, release, observations = asyncio.Event(), asyncio.Event(), []
    owner = None
    async def cleanup():
        await release.wait()
        await db.execute("INSERT INTO host_rollback_effects VALUES (292)")
    original = ownership.create_owned_task(cleanup(), retain_originals=True)
    manager._retain_quarantined_cleanup(name="Child", agent_id=agent.did, task=original)
    if not reconciled:
        manager._agents["Child"] = agent
    class Platform:
        def get_agent(self, name):
            return None
        async def terminate_child(self, parent, child, **kwargs):
            raise prior.error
    platform = Platform()
    async def reconcile(parent, child, *, withdrawals=None):
        kwargs = {} if withdrawals is None else {"withdrawals": withdrawals}
        return await manager.await_child_termination_reconciliation(parent, child, **kwargs)
    if entry == "instance":
        platform.await_child_termination_reconciliation = reconcile
    else:
        async def class_reconcile(self, parent, child, *, withdrawals=None):
            return await reconcile(parent, child, withdrawals=withdrawals)
        Platform.await_child_termination_reconciliation = class_reconcile
    actual_join = ownership.await_owned_task
    async def observed(task, *args, **kwargs):
        if task is original and asyncio.current_task() is owner:
            entered.set()
        outcome = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is owner and outcome.cancellation is not None:
            observations.append(OriginalFailure.capture(outcome.cancellation))
        return outcome
    monkeypatch.setattr(ownership, "await_owned_task", observed)
    lifecycle = SpawnedAgentLifecycle(platform)
    tracked = _TrackedChild("Child", agent.did, "did:test:declared-parent", SpawnMode.PERSISTENT, 0, "declared TTL observer")
    lifecycle._tracked["Child"] = tracked
    owner = ownership.create_owned_task(lifecycle._terminate_and_cleanup(
        "Child", SpawnStatus.TIMED_OUT, _terminal_descendant_fence_active=True), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual successful TTL reconciliation observer withdrew")
        await asyncio.sleep(0)
        release.set()
        outcome = await actual_join(owner)
        source.assert_retained(outcome.error)
        prior.assert_retained(outcome.error)
        assert len(observations) == 1
        observations[0].assert_retained(outcome.error)
        assert (lifecycle._tracked.get("Child") is None) is reconciled
        assert original.done() and await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    finally:
        release.set()
        await actual_join(original)
        await actual_join(owner)
        manager._agents.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["quarantine", "runtime", "budget", "descendant", "ttl"])
@pytest.mark.parametrize("through_cause", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_raised_first_join_cannot_abandon_later_original_owner(
    native_pg, onboarding_owner, monkeypatch, surface, through_cause, source_cancel,
):
    from kestrel_sovereign.features.isolated_runtime import RuntimeNamespaceCleanupOutcome
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    error = BaseExceptionGroup("whole first join source", [UnexpectedControl("original sibling")]
                              if through_cause else [source.error, UnexpectedControl("original sibling")])
    if through_cause:
        error.__cause__ = source.error
    group = OriginalFailure.capture(error)
    first_release, second_release = asyncio.Event(), asyncio.Event()
    first_observed, second_observed = asyncio.Event(), asyncio.Event()
    originals, owner, witness = [], None, None
    async def first():
        await first_release.wait()
        await db.execute("INSERT INTO host_rollback_effects VALUES (261)")
        raise group.error
    async def second():
        await second_release.wait()
        await db.execute("INSERT INTO host_rollback_effects VALUES (262)")
        return RuntimeNamespaceCleanupOutcome.REMOVED
    if surface == "budget":
        originals = [manager._start_child_budget_release("First", release_operation=first).task,
                     manager._start_child_budget_release("Second", release_operation=second).task]
    else:
        originals = [ownership.create_owned_task(first(), retain_originals=True),
                     ownership.create_owned_task(second(), retain_originals=True)]
        for index, task in enumerate(originals):
            name = ["First", "Second"][index]
            if surface in ("quarantine", "ttl"):
                manager._retain_quarantined_cleanup(name="SameChild" if surface == "ttl" else name, agent_id=agent.did, task=task)
            elif surface == "runtime":
                manager._inflight_runtime_offboardings[id(task)] = manager_module.InflightRuntimeOffboarding(name, agent.did, None, task)
            else:
                manager._agent_operations[name] = SimpleNamespace(kind="direct-spawn-test", spawn_task=task,
                    canonical_name=name.casefold(), spawn_parent=agent, committed=False)
    actual_join = manager_module.await_lifecycle_task_completion
    async def observed(task, **kwargs):
        if task is originals[0]:
            first_observed.set()
        if task is originals[1]:
            second_observed.set()
        return await actual_join(task, **kwargs)
    monkeypatch.setattr(manager_module, "await_lifecycle_task_completion", observed)
    operation = (manager.begin_terminal_descendant_spawn_fence(agent.did) if surface == "descendant" else
                 manager.await_child_termination_reconciliation("did:test:declared-parent", "SameChild") if surface == "ttl" else
                 manager.drain_quarantined_shutdowns())
    owner = ownership.create_owned_task(operation, retain_originals=True)
    try:
        await asyncio.wait_for(first_observed.wait(), 5)
        first_release.set()
        witness = ownership.create_owned_task(second_observed.wait(), retain_originals=True)
        done, _ = await asyncio.wait({owner, witness}, timeout=5)
        assert done and second_observed.is_set(), "raised first completion skipped the independent second original join"
        assert not originals[1].done() and not owner.done(), "terminal owner escaped while the second original was still live"
        second_release.set()
        result = await ownership.await_owned_task(owner)
        source.assert_retained(result.error)
        group.assert_retained(result.error)
        assert all(task.done() for task in originals)
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 2
        assert not manager._quarantined_shutdown_reapers and not manager._inflight_runtime_offboardings
        assert not manager._inflight_removal_budget_releases
        assert not manager._terminal_descendant_spawn_fence_members
    finally:
        first_release.set()
        second_release.set()
        for task in originals:
            await ownership.await_owned_task(task)
        if witness is not None:
            witness.cancel()
            await ownership.await_owned_task(witness)
        # Only fixture records; allow the broken negative's terminal loop out.
        manager._quarantined_shutdown_reapers.clear()
        manager._inflight_runtime_offboardings.clear()
        manager._inflight_removal_budget_releases.clear()
        manager._inflight_removal_budget_releases_by_child.clear()
        manager._agent_operations.clear()
        await ownership.await_owned_task(owner)
