"""Training retains actual source cancellation and unregistered release.

Session hooks are declared SQLite-backed local component inputs, NOT cloud
billing, provider termination or authenticated provider acceptance.
"""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.features.training.adapters._session_training_lifecycle import (
    SessionProviderHooks, SessionTrainingLifecycle,
)
from kestrel_sovereign.features.training.types import TrainingConfig
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_authority_postgres import native_pg as _cleanup_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

cleanup_pg = _cleanup_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["opening", "unregistered"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_training_finite_boundary_keeps_original_native_sources(
    native_pg, cleanup_pg, tmp_path, monkeypatch, stage, source_cancel,
):
    opening = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    cleanup = OriginalFailure.capture(await _original_control(cleanup_pg[0], monkeypatch, source_cancel))
    database = await AsyncDatabase.sqlite(str(tmp_path / "local-training-original.db"))
    await database.execute("CREATE TABLE local_session (acquired INTEGER, released INTEGER)")
    await database.execute("INSERT INTO local_session VALUES (0,0)")
    session, releases = object(), []
    async def acquire(*_args):
        await database.execute("UPDATE local_session SET acquired=acquired+1")
        if stage == "opening":
            raise opening.error
        return session
    def validate(_session):
        raise opening.error
    async def release(_session):
        assert _session is session
        releases.append(asyncio.current_task())
        await database.execute("UPDATE local_session SET released=released+1")
        raise cleanup.error
    async def submit(*_args):
        pytest.fail("rejected local session reached submission")
    lifecycle = SessionTrainingLifecycle(SessionProviderHooks(
        provider_name="declared-local", display_name="Declared local session",
        acquire_session=acquire, session_id=lambda _: "original-local-session",
        validate_session=validate, submit_job=submit, release_session=release,
    ))
    owner = create_owned_task(lifecycle.start(job_id="original", companion_id="declared",
        trigger_word="local", config=TrainingConfig(), avatar_data=b"local"), retain_originals=True)
    try:
        outcome = await await_owned_task(owner)
        opening.assert_retained(outcome.error)
        assert await database.fetchval("SELECT acquired FROM local_session") == 1
        if stage == "unregistered":
            cleanup.assert_retained(outcome.error, resource=session, task=releases[0])
            record = lifecycle._records["original"]
            assert record.session is session and record.release_task is releases[0]
            assert record.release_task.done()
            repeated = await await_owned_task(create_owned_task(lifecycle.close(), retain_originals=True))
            opening.assert_retained(repeated.error)
            cleanup.assert_retained(repeated.error)
            assert len(releases) == 1, "native-unresolved session release was repeated"
            assert await database.fetchval("SELECT released FROM local_session") == 1
    finally:
        await await_owned_task(owner)
        await await_owned_task(create_owned_task(lifecycle.close(), retain_originals=True))
        await database.close()
