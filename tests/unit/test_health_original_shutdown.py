"""Health's actual native owners do not depend on fresh teardown observers."""

import asyncio
import sqlite3

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_commit_outcome
from kestrel_sovereign.features.health.feature import HealthFeature
from tests.unit.test_cleanup_task_admission import (
    _operation_name,
    _partial_native_agent,
)
from tests.unit.test_execution_custody_review43_native import carries


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["loop", "base"])
async def test_public_health_shutdown_keeps_actual_write_and_original_owners(
    tmp_path, monkeypatch, refusal
):
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    feature = HealthFeature(agent)
    await feature.initialize()
    background = feature._background_task
    entered = asyncio.Event()
    original = OSError("actual health persistence COMMIT ACK lost")
    native_commit = db.backend._connection.commit
    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )

    async def native_peer():
        await db.execute("CREATE TABLE original_health_peer (id INTEGER)")
        await db.execute("INSERT INTO original_health_peer VALUES (1)")
        entered.set()
        await asyncio.Event().wait()

    async def lose_original_commit_ack():
        await native_commit()
        raise original

    def refuse_observer(loop, coroutine, **kwargs):
        name = _operation_name(coroutine)
        target = (
            "HealthFeature._stop_background_loop"
            if refusal == "loop"
            else "Feature.shutdown"
        )
        if name == target:
            refused.append(coroutine)
            raise RuntimeError("unnecessary health cleanup observer Task refused")
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    peer = feature._track_owned_background_task(
        native_peer(), name="actual-health-peer"
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        with monkeypatch.context() as fault:
            fault.setattr(db.backend._connection, "commit", lose_original_commit_ack)
            with pytest.raises(BaseException) as write:
                await feature.run_once()
        assert execution_commit_outcome(write.value) == "unknown"
        assert carries(write.value, original)
        loop.set_task_factory(refuse_observer)
        caught = None
        try:
            await feature.shutdown()
        except BaseException as error:
            caught = error
        loop.set_task_factory(previous)
        assert not refused, "health teardown requires a fresh cleanup observer Task"
        assert peer.done() and background.done(), (
            "original health owners were not physically joined"
        )
        assert caught is not None and execution_commit_outcome(caught) == "unknown"
        assert carries(caught, original)
        with sqlite3.connect(db.backend.db_path) as independent:
            assert (
                independent.execute("SELECT COUNT(*) FROM health_log").fetchone()[0]
                == 1
            )
            assert (
                independent.execute(
                    "SELECT COUNT(*) FROM original_health_peer"
                ).fetchone()[0]
                == 1
            )
        assert not agent._runtime_publication_ready.is_set()
    finally:
        loop.set_task_factory(previous)
        for coroutine in refused:
            coroutine.close()
        for task in (peer, background):
            if not task.done():
                task.cancel()
            await await_owned_task(task)
        try:
            await feature.shutdown()
        except BaseException as error:
            assert execution_commit_outcome(error) == "unknown"
        await agent.shutdown()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
async def test_health_shutdown_joins_original_pending_native_commit_before_cancellation(
    tmp_path, monkeypatch, unknown
):
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    feature = HealthFeature(agent)
    await feature.initialize()
    background = feature._background_task
    entered, release = asyncio.Event(), asyncio.Event()
    original = OSError("held original health COMMIT acknowledgement lost")
    connection = db.backend._connection
    native_commit, commits = connection.commit, []

    async def hold_original_commit_ack():
        await native_commit()
        commits.append(connection)
        entered.set()
        await release.wait()
        if unknown:
            raise original

    run = stop = None
    try:
        with monkeypatch.context() as fault:
            fault.setattr(connection, "commit", hold_original_commit_ack)
            run = create_owned_task(feature.run_once(), retain_originals=True)
            await asyncio.wait_for(entered.wait(), 2)
            writes = tuple(feature._health_persist_tasks)
            assert len(writes) == 1 and not writes[0].done()
            stop = create_owned_task(feature.shutdown(), retain_originals=True)
            async with asyncio.timeout(2):
                while stop not in getattr(writes[0], "_kestrel_owned_join_waiters", ()):
                    await asyncio.sleep(0.01)
            for message in ("first health withdrawal", "second health withdrawal"):
                stop.cancel(message)
                await asyncio.sleep(0)
                assert not stop.done() and not writes[0].done()
            release.set()
            outcome, finished_run = (
                await await_owned_task(stop),
                await await_owned_task(run),
            )
        assert commits == [connection] and background.done() and writes[0].done()
        assert not feature._health_persist_tasks
        if unknown:
            assert execution_commit_outcome(outcome.error) == "unknown"
            assert carries(outcome.error, original)
            assert execution_commit_outcome(finished_run.error) == "unknown"
        else:
            assert isinstance(outcome.error, asyncio.CancelledError)
            assert finished_run.error is None
        with sqlite3.connect(db.backend.db_path) as independent:
            assert (
                independent.execute("SELECT COUNT(*) FROM health_log").fetchone()[0]
                == 1
            )
        assert not agent._runtime_publication_ready.is_set()
    finally:
        release.set()
        for task in (stop, run, background):
            if task is not None:
                await await_owned_task(task)
        try:
            await feature.shutdown()
        except BaseException as error:
            assert execution_commit_outcome(error) == "unknown"
        await agent.shutdown()
        await db.close()
