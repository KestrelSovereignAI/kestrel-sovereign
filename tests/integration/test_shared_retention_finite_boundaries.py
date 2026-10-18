"""Source-only shared latches must preserve the current finite outcome separately."""
import asyncio

import pytest
from kestrel_sdk.features.host_base import HostFeature

from kestrel_sovereign._async_ownership import OwnedTaskAdmission, await_owned_task, create_owned_task, owned_task_admission_error
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_native_source_error, execution_terminal_error
from kestrel_sovereign.host_features import SovereignHostContext, start_host_features, stop_host_features
from kestrel_sovereign.host_features.context import _retained_host_cleanup_error
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.integration.test_native_completion_vs_observation import observation_factory
from tests.utils.native_admission_fault import NativeAdmissionFault
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("physical_source", ["native", "ordinary"])
async def test_canonical_join_distinguishes_physical_source_from_completion_observation(
    native_pg, onboarding_owner, monkeypatch, eager, source_cancel, physical_source,
):
    source = OriginalFailure.capture(
        await _original_control(native_pg[0], monkeypatch, source_cancel)
        if physical_source == "native" else OSError("original ordinary physical failure")
    )
    _, _, db = onboarding_owner
    observation = OSError("actual completion callback removal failure with native physical error")
    attached, accepted = asyncio.Event(), []
    async def producer():
        await db.execute("INSERT INTO host_rollback_effects VALUES (365)")
        raise source.error
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    loop.set_task_factory(observation_factory(
        lambda operation: getattr(operation, "cr_code", None) is producer.__code__,
        observation, attached, accepted, previous, eager=eager,
    ))
    task = None
    try:
        task = create_owned_task(producer(), retain_originals=True)
        result = await await_owned_task(task)
        source.assert_retained(result.error)
        assert observation in tuple(_explicit_control_chain(result.error))
        shared = execution_native_source_error(result.error)
        if physical_source == "native":
            source.assert_retained(shared)
            assert observation not in tuple(_explicit_control_chain(shared))
        else:
            assert shared is None
        source.assert_retained(source.error)
        assert task is accepted[0] and task.done()
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=365") == 1
    finally:
        loop.set_task_factory(previous)
        if task is not None:
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ["reaper", "budget"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_manager_independent_completion_sources_do_not_mutate_each_other(
    native_pg, onboarding_owner, monkeypatch, writer, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    manager, _, db = onboarding_owner
    resource = SQLiteBackend(":memory:")
    await resource.connect()
    worker = getattr(resource._connection, "_thread", resource._connection)
    admission = OwnedTaskAdmission()
    refused = OSError("actual native closer factory refused")
    fault = NativeAdmissionFault(lambda op: getattr(op, "cr_code", None) is SQLiteBackend.close.__code__, refused, "refuse")
    tasks, finite = [], []
    try:
        with fault:
            try:
                create_owned_task(resource.close(), retirement_resource=resource, retain_originals=True, admission=admission)
            except BaseException as error:
                debt = owned_task_admission_error(resource, admission, error)
        fault.assert_terminal_unstarted()
        debt_snapshot = OriginalFailure.capture(debt)
        assert admission.task is None and worker.is_alive()
        for index, failure in enumerate((source.error, debt)):
            async def cleanup(index=index, failure=failure):
                await db.execute("INSERT INTO host_rollback_effects VALUES (?)", (366 + index,))
                raise failure
            if writer == "budget":
                task = manager._start_child_budget_release("source-completion-" + str(index), release_operation=cleanup).task
            else:
                task = create_owned_task(cleanup(), retain_originals=True)
                manager._retain_quarantined_cleanup(name="source-completion-" + str(index), agent_id="component-native", task=task)
            tasks.append(task)
            finite.append((await await_owned_task(task)).error)
            await asyncio.sleep(0)
        shared = manager._quarantined_shutdown_control
        source.assert_retained(shared)
        debt_snapshot.assert_retained(finite[-1])
        from kestrel_sovereign.execution_custody import execution_cleanup_debts
        assert any(value is resource and task is None for value, task in execution_cleanup_debts(shared))
        source.assert_retained(source.error)
        debt_snapshot.assert_retained(debt_snapshot.error)
        assert admission.task is None and resource._connection is not None and worker.is_alive()
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id IN (366,367)") == 2
        assert not manager._inflight_removal_budget_releases and not manager._quarantined_shutdown_reapers
    finally:
        for task in tasks:
            await await_owned_task(task)
        await resource.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["stop-hook", "startup-hook", "stop-deactivate"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_host_hook_current_withdrawal_is_finite_not_shared(
    native_pg, monkeypatch, entry, source_cancel,
):
    startup = entry == "startup-hook"
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    db = SQLiteBackend(":memory:")
    await db.connect()
    await db.execute("CREATE TABLE effects (id INTEGER)")
    ctx = SovereignHostContext(db=db)
    entered, release, withdrawals, calls = asyncio.Event(), asyncio.Event(), [], []
    class Peer(HostFeature):
        name = "independent-native-peer"
        async def on_host_stop(self, ctx):
            calls.append("peer")
            await db.execute("INSERT INTO effects VALUES (369)")
    class Failing(HostFeature):
        name = "finite-native-hook"
        async def on_host_start(self, ctx):
            if startup:
                raise source.error
        async def on_host_stop(self, ctx):
            calls.append("failing")
            await db.execute("INSERT INTO effects VALUES (368)")
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError as error:
                withdrawals.append(error)
                finite = execution_terminal_error(source.error, error, retain_originals=True, isolate_control_originals=True)
                if entry == "stop-deactivate":
                    self.deactivation_failure = finite
                else:
                    raise finite
    features = [Peer(), Failing()]
    if entry == "stop-deactivate":
        from kestrel_sovereign.host_features.runtime import _host_contribution_runtime
        runtime = _host_contribution_runtime(ctx)
        actual_deactivate = runtime.deactivate
        def deactivate(feature):
            result = actual_deactivate(feature)
            if feature is features[1]:
                raise feature.deactivation_failure
            return result
        monkeypatch.setattr(runtime, "deactivate", deactivate)
    if not startup:
        await start_host_features(features, ctx)
    owner = create_owned_task(start_host_features(features, ctx) if startup else stop_host_features(features, ctx), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual current host-hook cleanup caller withdrew")
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        chain = tuple(_explicit_control_chain(result.error))
        assert withdrawals and all(error in chain for error in withdrawals)
        shared = _retained_host_cleanup_error(ctx)
        source.assert_retained(shared)
        assert all(error not in tuple(_explicit_control_chain(shared)) for error in withdrawals)
        assert calls == ["failing", "peer"]
        assert await db.fetch_val("SELECT count(*) FROM effects") == 2
        with pytest.raises(BaseException) as later:
            await start_host_features(features, ctx)
        source.assert_retained(later.value)
        assert all(error not in tuple(_explicit_control_chain(later.value)) for error in withdrawals)
        source.assert_retained(source.error)
    finally:
        release.set()
        await await_owned_task(owner)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_borrowed_pending_join_keeps_core_observer_group_provenance(
    native_pg, onboarding_owner, monkeypatch, eager, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    _, _, db = onboarding_owner
    entered, release = asyncio.Event(), asyncio.Event()
    observer_entered, observer_release = asyncio.Event(), asyncio.Event()
    withdrawals, fired = [], []
    async def producer():
        await db.execute("INSERT INTO host_rollback_effects VALUES (380)")
        entered.set()
        await release.wait()
    class BorrowedTask(asyncio.Task):
        add_failed = False
        remove_failed = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not self.add_failed and getattr(callback, "__name__", "") == "mark_complete":
                self.add_failed = True
                fired.append("add")
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
        def remove_done_callback(self, callback):
            result = super().remove_done_callback(callback)
            if not self.remove_failed and getattr(callback, "__name__", "") == "mark_complete":
                self.remove_failed = True
                fired.append("remove")
                raise withdrawals[0]
            return result
    from tests.unit.test_host_context_native_cleanup import _actual_operation
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **kwargs):
        if getattr(_actual_operation(coroutine), "cr_code", None) is producer.__code__:
            return BorrowedTask(coroutine, loop=loop, eager_start=eager, **kwargs)
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    loop.set_task_factory(factory)
    task = create_owned_task(producer(), retain_originals=True)
    loop.set_task_factory(previous)
    async def observe():
        observer_entered.set()
        try:
            await observer_release.wait()
        except asyncio.CancelledError as error:
            withdrawals.append(error)
            await await_owned_task(task, defer_cancellation=False)
    owner = create_owned_task(observe(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(observer_entered.wait(), 5)
        owner.cancel("actual borrowed caller withdrawal before failed callback publication")
        result = await await_owned_task(owner)
        assert fired == ["add", "remove"] and withdrawals
        source.assert_retained(result.error)
        assert withdrawals[0] in tuple(_explicit_control_chain(result.error))
        shared = execution_native_source_error(result.error)
        source.assert_retained(shared)
        assert withdrawals[0] not in tuple(_explicit_control_chain(shared))
        assert not task.done() and task.cancelling() == 0
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=380") == 1
    finally:
        release.set()
        observer_release.set()
        await await_owned_task(owner)
        await await_owned_task(task)
