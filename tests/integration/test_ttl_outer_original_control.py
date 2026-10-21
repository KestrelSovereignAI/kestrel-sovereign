"""Actual outer TTL/reader/drain ownership preserves native terminal sources.

Real local PG receipts, SQLite effects, Task cancellation and Core locks.
Declared lifecycle protocols do not confer signed authority or whole-Agent READY.
"""
import asyncio

import pytest

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_native_source_error, execution_observation_error
from kestrel_sovereign.multi_agent import agent_manager as manager_module
from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle, SpawnMode, SpawnStatus, _TrackedChild
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


class UnexpectedControl(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_child_reader_keeps_native_source_harvested_before_its_snapshot(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    async def cleanup():
        await db.execute("INSERT INTO host_rollback_effects VALUES (311)")
        raise source.error
    original = ownership.create_owned_task(cleanup(), retain_originals=True)
    manager._retain_quarantined_cleanup(name="Child", agent_id=agent.did, task=original)
    try:
        await ownership.await_owned_task(original)
        await asyncio.sleep(0)
        assert not manager._quarantined_shutdown_reapers
        assert manager._unsafe_quarantined_shutdown_failures
        assert manager._quarantined_cleanup_name_is_reserved("child")
        owner = ownership.create_owned_task(manager.await_child_termination_reconciliation("did:test:parent", "Child"), retain_originals=True)
        result = await ownership.await_owned_task(owner)
        source.assert_retained(result.error)
        assert not manager._quarantined_shutdown_reapers
        assert len(manager._unsafe_quarantined_shutdown_failures) == 1
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    finally:
        await ownership.await_owned_task(original)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["quarantine", "runtime", "budget"])
async def test_drain_keeps_unexpected_original_after_joining_independent_peer(
    onboarding_owner, monkeypatch, surface,
):
    from kestrel_sovereign.features.isolated_runtime import RuntimeNamespaceCleanupOutcome
    manager, agent, db = onboarding_owner
    control = OriginalFailure.capture(UnexpectedControl("original nonnative process control"))
    releases = [asyncio.Event(), asyncio.Event()]
    observed = [asyncio.Event(), asyncio.Event()]
    originals = []
    async def first():
        await releases[0].wait()
        await db.execute("INSERT INTO host_rollback_effects VALUES (321)")
        raise control.error
    async def second():
        await releases[1].wait()
        await db.execute("INSERT INTO host_rollback_effects VALUES (322)")
        return RuntimeNamespaceCleanupOutcome.REMOVED
    if surface == "budget":
        originals = [manager._start_child_budget_release("First", release_operation=first).task,
                     manager._start_child_budget_release("Second", release_operation=second).task]
    else:
        originals = [ownership.create_owned_task(first(), retain_originals=True),
                     ownership.create_owned_task(second(), retain_originals=True)]
        for index, task in enumerate(originals):
            name = ["First", "Second"][index]
            if surface == "quarantine":
                manager._retain_quarantined_cleanup(name=name, agent_id=agent.did, task=task)
            else:
                manager._inflight_runtime_offboardings[id(task)] = manager_module.InflightRuntimeOffboarding(name, agent.did, None, task)
    actual_join = manager_module.await_lifecycle_task_completion
    async def join(task, **kwargs):
        if task in originals:
            observed[originals.index(task)].set()
        return await actual_join(task, **kwargs)
    monkeypatch.setattr(manager_module, "await_lifecycle_task_completion", join)
    owner = ownership.create_owned_task(manager.drain_quarantined_shutdowns(), retain_originals=True)
    witness = None
    try:
        await asyncio.wait_for(observed[0].wait(), 5)
        releases[0].set()
        witness = ownership.create_owned_task(observed[1].wait(), retain_originals=True)
        done, _ = await asyncio.wait({owner, witness}, timeout=5)
        assert done and observed[1].is_set() and not originals[1].done()
        releases[1].set()
        result = await ownership.await_owned_task(owner)
        control.assert_retained(result.error)
        assert not isinstance(result.error, Exception), "unexpected control became ordinary availability failure"
        assert all(task.done() for task in originals)
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 2
    finally:
        for release in releases:
            release.set()
        for task in originals:
            await ownership.await_owned_task(task)
        await ownership.await_owned_task(owner)
        if witness is not None:
            witness.cancel()
            await ownership.await_owned_task(witness)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["cold", "live"])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("caller_cancel", [False, True])
async def test_actual_timer_retains_native_failure_and_current_observation(
    native_pg, onboarding_owner, monkeypatch, surface, source_cancel, caller_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    lifecycle = SpawnedAgentLifecycle(manager)
    entered, release, observations, physical = asyncio.Event(), asyncio.Event(), [], []
    timer = None
    async def component(*args, **kwargs):
        physical.append(asyncio.current_task())
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError as error:
            observations.append(OriginalFailure.capture(error))
            await db.execute("INSERT INTO host_rollback_effects VALUES (331)")
            raise execution_observation_error(source.error, error)
        await db.execute("INSERT INTO host_rollback_effects VALUES (331)")
        raise source.error
    actual_owned = ownership.await_owned_task
    async def observe(task, *args, **kwargs):
        outcome = await actual_owned(task, *args, **kwargs)
        if asyncio.current_task() is timer and outcome.cancellation is not None:
            observations.append(OriginalFailure.capture(outcome.cancellation))
        return outcome
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    if surface == "cold":
        monkeypatch.setattr(lifecycle, "_retire_stopped_authority_at_expiry", component)
    else:
        tracked = _TrackedChild("Child", agent.did, "did:test:parent", SpawnMode.EPHEMERAL, 1, "declared original timer")
        lifecycle._tracked["Child"] = tracked
        monkeypatch.setattr(lifecycle, "_terminate_and_cleanup", component)
    # Deliberately exercise the production legacy raw Task shape, not only a
    # canonical wrapper which could hide a broken timer's own terminal result.
    timer = asyncio.create_task(lifecycle._ttl_monitor("Child", agent.did, 0))
    if surface == "cold":
        lifecycle._retain_cold_ttl_owner(("child", agent.did), timer)
    else:
        tracked.ttl_task = timer
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if caller_cancel:
            timer.cancel("actual original TTL owner withdrew after expiry")
            await asyncio.sleep(0)
        release.set()
        result = await actual_owned(timer)
        source.assert_retained(result.error)
        assert len(observations) == int(caller_cancel)
        for observation in observations:
            observation.assert_retained(result.error)
            assert observation.error not in tuple(_explicit_control_chain(execution_native_source_error(manager._quarantined_shutdown_control)))
        source.assert_retained(manager._quarantined_shutdown_control)
        assert timer.done() and all(task.done() for task in physical)
        assert not lifecycle._cold_ttl_tasks
        assert surface == "cold" or tracked.ttl_task is timer, "native failure acquired a replacement retry timer"
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    finally:
        release.set()
        await actual_owned(timer)
        retries = set(lifecycle._cold_ttl_tasks.values())
        retries.update(item.ttl_task for item in lifecycle._tracked.values() if item.ttl_task is not None)
        for task in retries:
            if task is not timer:
                task.cancel()
                await actual_owned(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["instance", "class"])
async def test_successful_manager_false_reader_keeps_actual_withdrawal_without_prior_failure(
    onboarding_owner, monkeypatch, entry,
):
    manager, agent, db = onboarding_owner
    entered, release, observations = asyncio.Event(), asyncio.Event(), []
    owner = None
    async def cleanup():
        await release.wait()
        await db.execute("INSERT INTO host_rollback_effects VALUES (341)")
        raise OSError("ordinary cleanup remains unsafe")
    original = ownership.create_owned_task(cleanup(), retain_originals=True)
    manager._retain_quarantined_cleanup(name="Child", agent_id=agent.did, task=original)
    manager._agents["Child"] = agent
    class Platform:
        def get_agent(self, name):
            return agent
        def get_children(self, parent):
            return ["Child"]
        async def terminate_child(self, *args, **kwargs):
            return True
    platform = Platform()
    async def reconcile(parent, child, *, withdrawals=None):
        return await manager.await_child_termination_reconciliation(parent, child, withdrawals=withdrawals)
    if entry == "instance":
        platform.await_child_termination_reconciliation = reconcile
    else:
        async def class_reconcile(self, parent, child, *, withdrawals=None):
            return await reconcile(parent, child, withdrawals=withdrawals)
        Platform.await_child_termination_reconciliation = class_reconcile
    actual_owned = ownership.await_owned_task
    async def observe(task, *args, **kwargs):
        if task is original and asyncio.current_task() is owner:
            entered.set()
        outcome = await actual_owned(task, *args, **kwargs)
        if asyncio.current_task() is owner and outcome.cancellation is not None:
            observations.append(OriginalFailure.capture(outcome.cancellation))
        return outcome
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    lifecycle = SpawnedAgentLifecycle(platform)
    tracked = _TrackedChild("Child", agent.did, "did:test:parent", SpawnMode.PERSISTENT, 0, "declared refusal")
    lifecycle._tracked["Child"] = tracked
    owner = ownership.create_owned_task(lifecycle._terminate_and_cleanup(
        "Child", SpawnStatus.TIMED_OUT, _terminal_descendant_fence_active=True), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual false-reader observer withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await actual_owned(owner)
        assert len(observations) == 1
        observations[0].assert_retained(result.error)
        assert lifecycle._tracked.get("Child") is tracked and not lifecycle._results
        assert original.done() and await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    finally:
        release.set()
        await actual_owned(original)
        await actual_owned(owner)
        manager._agents.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["exceptional-settlement", "final-settlement", "hook"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_lifecycle_settlement_and_hook_withdrawal_keep_earlier_native_original(
    native_pg, onboarding_owner, monkeypatch, phase, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    prior = OriginalFailure.capture(
        BaseExceptionGroup("original untyped native termination", [source.error, OSError("original sibling")])
        if phase == "exceptional-settlement" else manager_module.RuntimeOffboardingRetainedError(
            agent_name="Child", agent_id=agent.did, runtime_path=None, cause=source.error, cleanup_pending=True))
    entered, release, observations = asyncio.Event(), asyncio.Event(), []
    owner = None
    class Platform:
        def get_agent(self, name):
            return None
        def get_children(self, parent):
            return []
        async def terminate_child(self, *args, **kwargs):
            raise prior.error
        async def revoke_terminal_spawn_scheduler_authority(self, name, did):
            return await manager.revoke_terminal_spawn_scheduler_authority(name, did)
    lifecycle = SpawnedAgentLifecycle(Platform())
    tracked = _TrackedChild("Child", agent.did, "did:test:parent", SpawnMode.PERSISTENT, 0, "declared finalization")
    lifecycle._tracked["Child"] = tracked
    actual_acquire = manager._lock.acquire
    held = phase != "hook"
    if held:
        await actual_acquire()
    async def acquire():
        if held and asyncio.current_task() is owner:
            entered.set()
        try:
            return await actual_acquire()
        except asyncio.CancelledError as error:
            if asyncio.current_task() is owner:
                observations.append(OriginalFailure.capture(error))
            raise
    monkeypatch.setattr(manager._lock, "acquire", acquire)
    class Hooks:
        async def execute_hooks(self, *args):
            await db.execute("INSERT INTO host_rollback_effects VALUES (351)")
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError as error:
                observations.append(OriginalFailure.capture(error))
                raise
    if phase == "hook":
        lifecycle._hooks_manager = Hooks()
    owner = ownership.create_owned_task(lifecycle._terminate_and_cleanup(
        "Child", SpawnStatus.TIMED_OUT, _terminal_descendant_fence_active=True), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual outer lifecycle finalization withdrew")
        result = await ownership.await_owned_task(owner)
        source.assert_retained(result.error)
        prior.assert_retained(result.error)
        assert len(observations) == 1
        observations[0].assert_retained(result.error)
        if held:
            assert not lifecycle._results and tracked.result is None
        assert await native_pg[1].fetchval(f'SELECT count(*) FROM "{native_pg[2]}".effects WHERE id=231') == 1
    finally:
        release.set()
        if held:
            manager._lock.release()
        await ownership.await_owned_task(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["member", "cause", "prior-native"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_lifecycle_shutdown_sweeps_peers_before_whole_native_report(
    native_pg, onboarding_owner, monkeypatch, shape, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    ordinary = OriginalFailure.capture(OSError("earlier ordinary child terminal"))
    unexpected = OriginalFailure.capture(UnexpectedControl("original second child control"))
    group = BaseExceptionGroup("original whole child outcome", [source.error, unexpected.error]
                               if shape == "member" else [unexpected.error])
    if shape == "cause":
        group.__cause__ = source.error
    whole = OriginalFailure.capture(group)
    lifecycle = SpawnedAgentLifecycle(manager)
    for name in ("First", "Second", "Third"):
        lifecycle._tracked[name] = _TrackedChild(name, agent.did, "did:test:parent", SpawnMode.PERSISTENT, 0, "declared sweep")
    calls = []
    async def terminate(name, **kwargs):
        calls.append(name)
        await db.execute("INSERT INTO host_rollback_effects VALUES (?)", (360+len(calls),))
        if name == "First":
            raise source.error if shape == "prior-native" else ordinary.error
        if name == "Second":
            raise unexpected.error if shape == "prior-native" else whole.error
        return True
    monkeypatch.setattr(lifecycle, "terminate", terminate)
    owner = ownership.create_owned_task(lifecycle.shutdown(), retain_originals=True)
    result = await ownership.await_owned_task(owner)
    source.assert_retained(result.error)
    unexpected.assert_retained(result.error)
    if shape != "prior-native":
        ordinary.assert_retained(result.error)
        whole.assert_retained(result.error)
    assert calls == ["First", "Second", "Third"]
    assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["report", "terminate"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_public_finalizer_keeps_inner_native_beside_release_failure(
    native_pg, onboarding_owner, monkeypatch, entry, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    later = OriginalFailure.capture(OSError("original finalization bookkeeping failed"))
    lifecycle = SpawnedAgentLifecycle(manager)
    lifecycle._tracked["Child"] = _TrackedChild("Child", agent.did, "did:test:parent", SpawnMode.PERSISTENT, 0, "declared public finalizer")
    async def terminate(*args, **kwargs):
        await db.execute("INSERT INTO host_rollback_effects VALUES (371)")
        raise source.error
    actual_release = lifecycle._release_finalization
    def release(*args):
        actual_release(*args)
        raise later.error
    monkeypatch.setattr(lifecycle, "_terminate_and_cleanup", terminate)
    monkeypatch.setattr(lifecycle, "_release_finalization", release)
    owner = ownership.create_owned_task(lifecycle.report_result("Child") if entry == "report" else lifecycle.terminate("Child"), retain_originals=True)
    result = await ownership.await_owned_task(owner)
    source.assert_retained(result.error)
    later.assert_retained(result.error)
    assert not lifecycle._finalization_owner_counts
    assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
