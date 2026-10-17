"""Actual SQLite COMMIT retirement precedes scheduler Stop/effect publication."""
import asyncio
from datetime import datetime, timezone

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.features.scheduler.runner import ScheduledTask, SchedulerRunner
from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError, execution_commit_outcome
from tests.unit.test_scheduler_durability import _database


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [None, "unknown", "committed"])
async def test_stop_during_native_admission_commit_joins_original_ack_and_dispatches_nothing(tmp_path, monkeypatch, receipt):
    db = await _database(tmp_path / "scheduler.db")
    did = "did:test:native-scheduler-admission"
    async def effect(*args):
        raise AssertionError("cancelled admission dispatched an effect")
    runner = SchedulerRunner(db, did, effect)
    await runner._ensure_tables()
    now = datetime.now(timezone.utc).isoformat()
    task = ScheduledTask("original", did, "effect", "* * * * *", "{}", True, None, now, now)
    connection = db.backend._connection
    native_commit = connection.commit
    entered, release = asyncio.Event(), asyncio.Event()
    commits, effects = [], []
    async def held_native_commit():
        await native_commit()
        commits.append(connection)
        entered.set()
        await release.wait()
        if receipt is not None:
            raise ExecutionCommitOutcomeError(receipt)
    monkeypatch.setattr(connection, "commit", held_native_commit)
    async def admit():
        async with runner._active_dispatch_admission(task) as admitted:
            effects.append(admitted)
    original = create_owned_task(admit())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        original.cancel()
        done, _ = await asyncio.wait({original}, timeout=.05)
        assert not done, "scheduler discarded the original native admission COMMIT owner"
        original.cancel()
        release.set()
        outcome = await await_owned_task(original)
        if receipt is None:
            assert isinstance(outcome.error, asyncio.CancelledError)
        else:
            assert execution_commit_outcome(outcome.error) == receipt
        assert effects == [] and commits == [connection]
        monkeypatch.setattr(connection, "commit", native_commit)
        await db.execute("CREATE TABLE original_commit_joined(id INTEGER)")
    finally:
        release.set()
        await await_owned_task(original)
        monkeypatch.setattr(connection, "commit", native_commit)
        await db.close()
