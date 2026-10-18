"""Finite enclosing lifecycle boundaries keep native originals immutable.

Actual PostgreSQL COMMIT-ACK sources and SQLite worker effects; declared phase
callbacks are not whole-Agent, authority/signature, provider or funds acceptance.
"""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from kestrel_sovereign import server
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task, owned_task_error
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_execution_custody_review18_pg import setup_delivery, join_owned
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.integration.test_native_completion_vs_observation import observation_factory
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


class OrderedRepairs(set):
    """Deterministic order with actual producer-style set/discard ownership."""
    def __init__(self, tasks=()):
        super().__init__(tasks)
        self.order = list(tasks)

    def add(self, task):
        super().add(task)
        self.order.append(task)

    def __iter__(self):
        return iter(tuple(task for task in self.order if task in self))


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["repairs", "host-features", "server-phases"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_ordinary_prefix_is_not_rewritten_before_later_native_control(
    native_pg, onboarding_owner, monkeypatch, surface, source_cancel,
):
    _, _, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    ordinary = [OriginalFailure.capture(OSError("original ordinary first")),
                OriginalFailure.capture(ValueError("original ordinary second"))]
    originals = [*ordinary, source]
    calls, tasks, cleanup = [], [], None
    def operation(index):
        async def effect(*args, **kwargs):
            calls.append(index)
            await db.execute("INSERT INTO host_rollback_effects VALUES (?)", (101 + index,))
            raise originals[index].error
        return effect
    if surface == "repairs":
        agent, dispatcher, _, _ = await setup_delivery(native_pg)
        cleanup = (agent, dispatcher)
        tasks = [create_owned_task(operation(index)(), retain_originals=True) for index in range(3)]
        dispatcher._post_commit_reservation_repairs = OrderedRepairs(tasks)
        for task in tasks:
            task.add_done_callback(dispatcher._post_commit_reservation_repairs.discard)
        op = dispatcher._drain_post_commit_reservation_repairs()
    elif surface == "host-features":
        from kestrel_sovereign import host_features
        app = FastAPI()
        app.state.host_context = object()
        def unmount_first(app):
            calls.append(0)
            raise ordinary[0].error
        def unmount_second(app):
            calls.append(1)
            raise ordinary[1].error
        monkeypatch.setattr(host_features, "unmount_host_features", unmount_first)
        monkeypatch.setattr(server, "_unmount_feature_routers", unmount_second)
        monkeypatch.setattr(server, "_unmount_feature_ui_assets", lambda app: None)
        monkeypatch.setattr(host_features, "close_host_context", operation(2))
        op = server._shutdown_host_features(app)
    else:
        app = FastAPI()
        for name in ("_shutdown_host_scheduler", "_shutdown_stop_cleanup", "_shutdown_server_agents",
                     "_shutdown_host_features", "_shutdown_distributed_invocations",
                     "_shutdown_stop_receipts", "_shutdown_shared_agent_postgres_backend", "_shutdown_phoenix"):
            async def nothing(*args, **kwargs):
                pass
            monkeypatch.setattr(server, name, nothing)
        monkeypatch.setattr(server, "_shutdown_host_scheduler", operation(0))
        monkeypatch.setattr(server, "_shutdown_stop_cleanup", operation(1))
        monkeypatch.setattr(server, "_shutdown_phoenix", operation(2))
        op = server._shutdown_server_resources(app)
    owner = create_owned_task(op, retain_originals=True)
    try:
        result = await await_owned_task(owner)
        failure = result.result[1] if surface == "server-phases" and result.error is None else result.error
        for original in originals:
            original.assert_retained(failure)
        assert calls == [0, 1, 2]
        assert all(task.done() for task in tasks)
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == (1 if surface == "host-features" else 3)
    finally:
        await await_owned_task(owner)
        for task in tasks:
            await await_owned_task(task)
        if cleanup is not None:
            dispatcher._post_commit_reservation_repairs.clear()
            await join_owned(*cleanup)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["reaper", "budget"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_manager_drain_keeps_observation_only_native_control(
    native_pg, onboarding_owner, monkeypatch, surface, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    entered, release, attached = asyncio.Event(), asyncio.Event(), asyncio.Event()
    accepted, owner = [], None
    async def native():
        await db.execute("INSERT INTO host_rollback_effects VALUES (110)")
        entered.set()
        await release.wait()
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    loop.set_task_factory(observation_factory(
        lambda op: getattr(op, "cr_code", None) is native.__code__,
        source.error, attached, accepted, previous, eager=False,
    ))
    try:
        if surface == "budget":
            task = manager._start_child_budget_release("finite-observer", release_operation=native).task
        else:
            task = create_owned_task(native(), retain_originals=True)
            manager._retain_quarantined_cleanup(name="finite-observer", agent_id=agent.did, task=task)
        loop.set_task_factory(previous)
        owner = create_owned_task(manager.drain_quarantined_shutdowns(), retain_originals=True)
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(attached.wait(), 5)
        release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert task.done() and owned_task_error(task) is None
        assert vars(manager).get("_quarantined_shutdown_control") is None
        assert not manager._lock.locked() and not manager._quarantined_shutdown_drain_lock.locked()
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    finally:
        release.set()
        loop.set_task_factory(previous)
        if owner is not None:
            await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_manager_unseal_does_not_replace_prior_native_reconciliation(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    manager, _, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    ordinary = OriginalFailure.capture(OSError("original unseal observation failed"))
    original_seal = manager._set_quarantined_shutdown_handoffs_sealed
    async def reconcile():
        await db.execute("INSERT INTO host_rollback_effects VALUES (111)")
        raise source.error
    async def seal(value, **kwargs):
        result = await original_seal(value, **kwargs)
        if not value:
            await db.execute("INSERT INTO host_rollback_effects VALUES (112)")
            raise ordinary.error
        return result
    monkeypatch.setattr(manager, "_prune_all_fully_removed_child_tracking", reconcile)
    monkeypatch.setattr(manager, "_set_quarantined_shutdown_handoffs_sealed", seal)
    result = await await_owned_task(create_owned_task(manager.drain_quarantined_shutdowns(), retain_originals=True))
    source.assert_retained(result.error)
    ordinary.assert_retained(result.error)
    assert not manager._quarantined_shutdown_handoffs_sealed
    assert not manager._quarantined_shutdown_drain_lock.locked()
    assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == 2

@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["late-accepted", "native-before-repairs"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_reservation_repair_drain_owns_late_admissions_and_outer_native_prefix(
    native_pg, onboarding_owner, monkeypatch, surface, source_cancel,
):
    from kestrel_sovereign import _async_ownership as ownership
    from kestrel_sovereign.execution_custody import _explicit_control_chain
    _, _, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    ordinary = [OriginalFailure.capture(OSError("original repair one")),
                OriginalFailure.capture(ValueError("original repair two"))]
    agent, dispatcher, _, _ = await setup_delivery(native_pg)
    entered, release, late_entered, late_release = (asyncio.Event() for _ in range(4))
    drain_entered = asyncio.Event()
    repairs, tasks, withdrawals, owner = OrderedRepairs(), [], [], None
    actual_join = ownership.await_owned_task
    async def observe(task, *args, **kwargs):
        if asyncio.current_task() is owner and task is first_task:
            drain_entered.set()
        result = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is owner and result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    async def first():
        await db.execute("INSERT INTO host_rollback_effects VALUES (150)")
        entered.set()
        await release.wait()
        if surface == "late-accepted":
            publish()
        raise ordinary[0].error
    async def later():
        await db.execute("INSERT INTO host_rollback_effects VALUES (151)")
        late_entered.set()
        await late_release.wait()
        raise source.error if surface == "late-accepted" else ordinary[1].error
    def publish():
        task = create_owned_task(later(), retain_originals=True)
        repairs.add(task)
        tasks.append(task)
        task.add_done_callback(repairs.discard)
    first_task = create_owned_task(first(), retain_originals=True)
    repairs.add(first_task)
    tasks.append(first_task)
    first_task.add_done_callback(repairs.discard)
    dispatcher._post_commit_reservation_repairs = repairs
    if surface == "late-accepted":
        op = dispatcher._drain_post_commit_reservation_repairs()
    else:
        async def writer():
            await db.execute("INSERT INTO host_rollback_effects VALUES (152)")
            raise source.error
        task = create_owned_task(writer(), retain_originals=True)
        dispatcher._outcome_log_tasks.add(task)
        def retire_writer(completed):
            dispatcher._outcome_log_tasks.discard(completed)
            dispatcher._retain_durable_writer_control(owned_task_error(completed))
        task.add_done_callback(retire_writer)
        tasks.append(task)
        publish()
        op = dispatcher._complete_durable_shutdown()
    owner = create_owned_task(op, retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        # Hold both original repairs until the enclosing owner has captured
        # its repair snapshot, rather than demanding discarded past history.
        await asyncio.wait_for(drain_entered.wait(), 5)
        owner.cancel("actual repair drain observer withdrew")
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(late_entered.wait(), 5)
        assert not owner.done(), "drain abandoned an accepted late repair after an ordinary failure"
        late_release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        ordinary[0].assert_retained(result.error)
        if surface == "native-before-repairs":
            ordinary[1].assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert all(task.done() for task in tasks)
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == (2 if surface == "late-accepted" else 3)
    finally:
        release.set()
        late_release.set()
        await await_owned_task(owner)
        for task in tasks:
            await await_owned_task(task)
        await join_owned(agent, dispatcher)
