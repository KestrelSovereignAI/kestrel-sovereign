"""Accepted training operations remain owned through callback-publication failure.

Hooks are declared local SQLite components, NOT authenticated cloud/billing
acceptance. Original control receipts come from actual localhost PostgreSQL.
"""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.features.training.adapters._session_training_lifecycle import SessionProviderHooks, SessionTrainingLifecycle, ReleaseIntent
from kestrel_sovereign.features.training.types import TrainingConfig
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_authority_postgres import native_pg as _cleanup_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure

cleanup_pg = _cleanup_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["submission", "compensation", "release"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_training_publication_keeps_accepted_task_phase_and_original_outcomes(
    native_pg, cleanup_pg, tmp_path, monkeypatch, stage, source_cancel,
):
    opening = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    physical = OriginalFailure.capture(await _original_control(cleanup_pg[0], monkeypatch, source_cancel))
    database = await AsyncDatabase.sqlite(str(tmp_path / "owned-training-publication.db"))
    await database.execute("CREATE TABLE local_effects (kind TEXT)")
    session, entered, release = object(), asyncio.Event(), asyncio.Event()
    accepted, hooks, fired = [], [], []
    async def acquire(*args):
        return session
    async def fault(kind):
        hooks.append((kind, asyncio.current_task()))
        await database.execute("INSERT INTO local_effects VALUES (?)", (kind,))
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        raise physical.error
    async def submit(*args):
        if stage == "submission":
            return await fault("submission")
        return "declared-provider-job"
    async def compensate(*args):
        return await fault("compensation")
    async def release_session(_session):
        assert _session is session
        if stage in ("release", "submission"):
            return await fault("release")
        await database.execute("INSERT INTO local_effects VALUES ('release')")
    lifecycle = SessionTrainingLifecycle(SessionProviderHooks(
        provider_name="declared-local", display_name="Declared local training",
        acquire_session=acquire, session_id=lambda _: "original-local-session",
        submit_job=submit, release_session=release_session,
        cancel_provider_job=compensate if stage == "compensation" else None,
    ))
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    target = {"submission": "_run_submission", "compensation": "_run_compensation", "release": "_run_release"}[stage]
    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *args, **kwargs):
            if not fired and getattr(callback, "__name__", "") == "_observe_outcome":
                fired.append(True)
                super().add_done_callback(callback, *args, **kwargs)
                raise opening.error
            return super().add_done_callback(callback, *args, **kwargs)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname.endswith("SessionTrainingLifecycle." + target):
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    async def start():
        return await lifecycle.start(job_id="original", companion_id="declared",
            trigger_word="local", config=TrainingConfig(), avatar_data=b"local")
    owner = None
    try:
        if stage != "submission":
            record = await start()
            await await_owned_task(record.submission_task)
        loop.set_task_factory(factory)
        operation = start() if stage == "submission" else (
            lifecycle.stop_submission("original", reason="declared stop") if stage == "compensation"
            else lifecycle.release("original", ReleaseIntent.CANCEL))
        owner = create_owned_task(operation, retain_originals=True)
        await asyncio.wait_for(entered.wait(), 5)
        returned_early = owner.done()
        release.set()
        result = await await_owned_task(owner)
        loop.set_task_factory(previous)
        for task in accepted:
            await await_owned_task(task)
        assert fired and len(accepted) == 1 and accepted[0].done()
        assert not returned_early, "training boundary abandoned accepted callback-failed original"
        opening.assert_retained(result.error)
        physical.assert_retained(result.error)
        if stage != "release":
            await await_owned_task(create_owned_task(lifecycle.close(), retain_originals=True))
        # Submission admission failed before its first turn. It must be
        # cancelled/closed, not started merely to obtain a test outcome.
        assert await database.fetchval("SELECT count(*) FROM local_effects WHERE kind=?", (stage,)) == (0 if stage == "submission" else 1)
        if stage == "release":
            await await_owned_task(create_owned_task(lifecycle.close(), retain_originals=True))
            assert await database.fetchval("SELECT count(*) FROM local_effects WHERE kind='release'") == 1
        else:
            assert await database.fetchval("SELECT count(*) FROM local_effects WHERE kind='release'") == 1
    finally:
        loop.set_task_factory(previous)
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        await await_owned_task(create_owned_task(lifecycle.close(), retain_originals=True))
        await database.close()
