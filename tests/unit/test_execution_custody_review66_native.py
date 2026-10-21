"""Native status COMMIT acknowledgement is not cancelled by observer Stop."""
import asyncio

import pytest
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_commit_outcome
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("commit_index", [1, 2, 3])
async def test_native_status_observer_stop_joins_original_sqlite_commit(tmp_path, monkeypatch, commit_index):
    backend = SQLiteBackend(str(tmp_path / "status.db"))
    await backend.connect()
    db = AsyncDatabase(backend)
    async def no_work(*args):
        raise AssertionError("telemetry cannot execute scheduled work")
    runner = SchedulerRunner(db, "native-original", no_work)
    await runner._ensure_tables()
    entered, release = asyncio.Event(), asyncio.Event()
    native_commit = backend._connection.commit
    commits = []
    async def held_commit():
        await native_commit()
        commits.append(True)
        if len(commits) == commit_index:
            entered.set()
            await release.wait()
    monkeypatch.setattr(backend._connection, "commit", held_commit)
    publisher = create_owned_task(runner._publish_runtime_status("running"))
    runner._runtime_status_publish_task = publisher
    stopping = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        stopping = create_owned_task(runner._join_runtime_status_publish_task(
            deadline=asyncio.get_running_loop().time()))
        done, _ = await asyncio.wait({stopping}, timeout=.05)
        assert not done, "observer Stop discarded original native COMMIT acknowledgement"
        release.set()
        result = await await_owned_task(stopping)
        assert execution_commit_outcome(result.error) is None
        assert runner._execution_control_error is None and runner._readiness_failure is None
        assert runner._runtime_status_publish_task is None
        assert len(commits) == commit_index
    finally:
        release.set()
        await await_owned_task(publisher)
        if stopping is not None:
            await await_owned_task(stopping)
        await db.close()
