"""Actual SQLite writer handoff and delivery/health idle-restart consumers."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionCustody, ProcessRuntimeExecutionFence, bind_execution_runtime,
    execution_commit_outcome, is_execution_control_error,
)
from kestrel_sovereign.services.payer_resolver import open_host_db
from kestrel_sovereign.storage.async_database import AsyncDatabase


async def host_accessor(tmp_path):
    path = tmp_path / "agent_data"
    path.mkdir()
    seed = await AsyncDatabase.sqlite(str(path / "host.db"))
    await seed.close()
    owner = SimpleNamespace(_execution_custody=ExecutionCustody(ProcessRuntimeExecutionFence("original-host")))
    with bind_execution_runtime(owner):
        db = await open_host_db(storage_path=path / "agent.db")
    return owner, db, path / "host.db"


async def verify_no_effects(path):
    db = await AsyncDatabase.sqlite(str(path))
    try:
        assert await db.fetchone("SELECT COUNT(*) FROM service_providers WHERE id='queued'") == (0,)
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["execute", "batch", "script"])
async def test_host_write_waiting_for_real_sqlite_writer_lock_cannot_outlive_runtime(tmp_path, operation):
    owner, db, path = await host_accessor(tmp_path)
    sql = "INSERT INTO service_providers (id,name) VALUES ('queued','queued')"
    backend = db.backend
    await backend._write_lock.acquire()
    task = None
    try:
        with bind_execution_runtime(owner):
            if operation == "execute":
                coro = db.execute(sql)
            elif operation == "batch":
                coro = db.execute_many(sql, [()])
            else:
                coro = db.execute_script(sql)
            task = asyncio.create_task(coro)
            await asyncio.sleep(0)
            assert not task.done()
            assert len(backend._write_lock._waiters) == 1
            owner._execution_custody.fence.retire()
            backend._write_lock.release()
            result = (await asyncio.gather(task, return_exceptions=True))[0]
            assert isinstance(result, BaseException) and is_execution_control_error(result)
            assert execution_commit_outcome(result) is None, "queued denial arrived only after COMMIT"
    finally:
        if backend._write_lock.locked():
            backend._write_lock.release()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await db.close()
    await verify_no_effects(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["execute", "batch"])
async def test_host_native_statement_receipt_is_rechecked_before_commit(tmp_path, monkeypatch, operation):
    owner, db, path = await host_accessor(tmp_path)
    connection = db.backend._connection
    method = "execute" if operation == "execute" else "executemany"
    original = getattr(connection, method)

    async def statement_then_retire(*args, **kwargs):
        result = await original(*args, **kwargs)
        owner._execution_custody.fence.retire()
        return result

    monkeypatch.setattr(connection, method, statement_then_retire)
    sql = "INSERT INTO service_providers (id,name) VALUES ('queued','queued')"
    try:
        with bind_execution_runtime(owner):
            with pytest.raises(BaseException) as caught:
                if operation == "execute":
                    await db.execute(sql)
                else:
                    await db.execute_many(sql, [()])
            assert is_execution_control_error(caught.value)
            assert execution_commit_outcome(caught.value) is None
    finally:
        await db.close()
    await verify_no_effects(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("feature_name", ["delivery", "health"])
async def test_actual_booted_idle_restart_allows_resident_delivery_and_health(tmp_path, feature_name):
    from kestrel_sovereign.features.delivery.feature import DeliveryFeature
    from kestrel_sovereign.features.health.feature import HealthFeature
    from kestrel_sovereign.features.restart_coordinator.feature import RestartCoordinatorFeature
    from tests.unit.test_restart_idle_gate_booted_agent import _booted_idle_agent

    async with _booted_idle_agent(tmp_path) as owner:
        # Boot's unrelated storage collaborator is a double. Supply a real
        # SQLite queue database through the production storage resolver;
        # neither the resident loop nor the idle gate is mocked here.
        original_storage = owner._raw_storage
        database = await AsyncDatabase.sqlite(str(tmp_path / "delivery.db"))
        owner._raw_storage = SimpleNamespace(db=database)
        feature = DeliveryFeature(owner) if feature_name == "delivery" else HealthFeature(owner)
        work = None
        entered = asyncio.Event()

        async def real_work():
            entered.set()
            await asyncio.Event().wait()

        try:
            await feature.initialize()
            await asyncio.sleep(0)
            resident = feature._queue._task if feature_name == "delivery" else feature._background_task
            assert resident in owner._background_tasks and not resident.done()
            coordinator = RestartCoordinatorFeature(owner)
            assert coordinator._agent_appears_idle()["idle"], "new resident permanently blocks idle restart"
            work = owner._track_background_task(real_work(), name="actual-delivery-user-work")
            await entered.wait()
            assert not coordinator._agent_appears_idle()["idle"], "ordinary effect child was excluded with residents"
            if feature_name == "delivery":
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                await asyncio.sleep(0)
                delivered, release_delivery = asyncio.Event(), asyncio.Event()

                async def deliver(_channel, _recipient, _content):
                    from kestrel_sovereign.features.delivery.models import DeliveryResult
                    delivered.set()
                    await release_delivery.wait()
                    return DeliveryResult(success=True)

                feature._queue._deliver = deliver
                await feature.enqueue_message("synthetic", "test-only", {"body": "fixture"})
                # Execute the actual next polling iteration now, not after a
                # ten-second sleep. Real process_pending creates the attempt.
                poll = asyncio.create_task(feature._queue.process_pending())
                try:
                    await asyncio.wait_for(delivered.wait(), 5)
                    state = coordinator._agent_appears_idle()
                    assert not state["idle"] and "delivery-attempt:" in state["reason"]
                finally:
                    release_delivery.set()
                    await poll
                await asyncio.sleep(0)
                assert coordinator._agent_appears_idle()["idle"]
        finally:
            if work is not None:
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
            await feature.shutdown()
            owner._raw_storage = original_storage
            await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocker", ["none", "same-name", "sibling-agent", "other-schedule"])
async def test_real_scheduler_restart_excludes_only_its_original_live_occurrence(tmp_path, blocker):
    from kestrel_sovereign.features.restart_coordinator.feature import RestartCoordinatorFeature
    from kestrel_sovereign.features.scheduler.runner import (
        SchedulerRunner, capture_scheduler_execution_scope,
        get_current_scheduler_occurrence_task,
    )
    from tests.unit.test_restart_idle_gate_booted_agent import _booted_idle_agent
    from tests.unit.test_scheduler_durability import _database, _seed_due
    from tests.utils.aiosqlite_workers import normal_close_schedule
    from tests.utils.scheduler_ticks import tick_and_settle

    async with _booted_idle_agent(tmp_path) as owner:
        database = await _database(tmp_path / "restart-schedule.db")
        coordinator = RestartCoordinatorFeature(owner)
        req = SimpleNamespace(policy="idle_agents_only", requested_at=datetime.now(timezone.utc).isoformat())
        observed = []
        scopes = []
        original_tasks = []
        blocker_task = None

        async def inspect_from_dispatch_child():
            # Actual runner preparation/effect paths cross task boundaries.
            # This additionally exercises an awaited tool-dispatch child.
            original = get_current_scheduler_occurrence_task(
                agent_id=owner.did, task_name="restart_coordinator",
            )
            original_tasks.append(original)
            if original is not None:
                assert original in owner._background_tasks and not original.done()
                assert original is not asyncio.current_task()
            assert not coordinator._agent_appears_idle()["idle"]
            scopes.append(capture_scheduler_execution_scope())
            observed.append(coordinator._evaluate_safety(req, database_now=datetime.now(timezone.utc)))
            return "inspected"

        async def executor(_name, _args):
            return await asyncio.create_task(inspect_from_dispatch_child())

        runner = SchedulerRunner(
            database, owner.did, executor, owner_id="restart-inspector",
            occurrence_task_factory=owner._track_background_task,
        )
        try:
            await runner._ensure_tables()
            await _seed_due(
                database, task_id="restart-row", agent_id=owner.did,
                task_name="ordinary-task" if blocker == "other-schedule" else "restart_coordinator",
            )
            if blocker in {"same-name", "sibling-agent"}:
                blocker_task = owner._track_background_task(
                    asyncio.Event().wait(), name="scheduler-occurrence:restart-row",
                )
                if blocker == "sibling-agent":
                    # The owning task remains original; only the sibling
                    # actually exposes the unrelated second task as work.
                    owner._background_tasks.discard(blocker_task)
                    sibling = SimpleNamespace(
                        did="did:test:sibling", _active_request_ids=set(),
                        _background_tasks={blocker_task},
                    )
                    owner._cohosted_agents_provider = lambda: [owner, sibling]
            await tick_and_settle(runner)
            assert len(observed) == 1
            assert observed[0]["safe"] is (blocker == "none"), observed
            if blocker == "none":
                assert original_tasks[0] is not None
            assert scopes and not scopes[0].active and scopes[0].occurrence_task is None
            assert await database.fetchone(
                "SELECT status FROM task_execution_log WHERE task_id='restart-row'"
            ) == ("success",)
        finally:
            if blocker_task is not None:
                blocker_task.cancel()
                await asyncio.gather(blocker_task, return_exceptions=True)
            await runner.stop()
            # This test asserts restart ownership, not WAL/fsync throughput.
            # Retire this exact real connection/worker before returning while
            # keeping production close deadlines covered by their own tests.
            backend = database.backend
            connection = backend._ensure_connected()
            async with normal_close_schedule() as finish:
                await finish(
                    connection, database.close(),
                    retained_closes=backend._retired_connection_closes,
                )
