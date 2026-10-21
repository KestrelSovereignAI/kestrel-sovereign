"""Operational native failure cannot fence an ordinary physical release retry.

Hooks model a local session with SQLite effects, not actual provider billing.
"""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.features.training.adapters._session_training_lifecycle import (
    ReleaseIntent, SessionJobPhase, SessionProviderHooks, SessionTrainingLifecycle,
)
from kestrel_sovereign.features.training.types import TrainingConfig
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["submission", "compensation"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_operational_native_failure_does_not_prevent_ordinary_release_retry(
    native_pg, tmp_path, monkeypatch, stage, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    database = await AsyncDatabase.sqlite(str(tmp_path / "release-retry.db"))
    await database.execute("CREATE TABLE effects (attempt INTEGER)")
    session, attempts = object(), []
    ordinary = OSError("declared first physical release failed before local acknowledgement")
    async def acquire(*args):
        return session
    async def submit(*args):
        if stage == "submission":
            raise source.error
        return "declared-job"
    async def compensate(*args):
        raise source.error
    async def release_session(current):
        assert current is session
        attempts.append(asyncio.current_task())
        await database.execute("INSERT INTO effects VALUES (?)", (len(attempts),))
        if len(attempts) == 1:
            raise ordinary
    lifecycle = SessionTrainingLifecycle(SessionProviderHooks(
        provider_name="local", display_name="Local release retry",
        acquire_session=acquire, session_id=lambda _: "declared-session",
        submit_job=submit, release_session=release_session,
        cancel_provider_job=compensate if stage == "compensation" else None,
    ))
    try:
        record = await lifecycle.start(job_id="original", companion_id="declared",
            trigger_word="local", config=TrainingConfig(), avatar_data=b"local")
        await await_owned_task(record.submission_task)
        first = await await_owned_task(create_owned_task(lifecycle.release("original", ReleaseIntent.CANCEL), retain_originals=True))
        source.assert_retained(first.error)
        assert ordinary in tuple(_explicit_control_chain(first.error))
        assert record.phase is SessionJobPhase.RELEASE_FAILED and not record.session_released
        assert record.native_release_control is None, "operational native source was mistaken for physical release uncertainty"
        second = await await_owned_task(create_owned_task(lifecycle.release("original", ReleaseIntent.CANCEL), retain_originals=True))
        source.assert_retained(second.error)
        assert record.session_released and record.phase is SessionJobPhase.RELEASED
        assert len(attempts) == 2 and all(task.done() for task in attempts)
        assert await database.fetchval("SELECT count(*) FROM effects") == 2
    finally:
        await await_owned_task(create_owned_task(lifecycle.close(), retain_originals=True))
        await database.close()
