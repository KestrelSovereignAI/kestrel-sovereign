"""Training join observations retain actual control without provider requests.

The session hooks are declared local component inputs backed by disposable
SQLite state; these are NOT cloud billing/provider retirement acceptance.
"""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.features.training.adapters import _session_training_lifecycle as lifecycle_module
from kestrel_sovereign.features.training.adapters._session_training_lifecycle import SessionProviderHooks, SessionTrainingLifecycle
from kestrel_sovereign.features.training.types import TrainingConfig
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["opening", "release"])
@pytest.mark.parametrize("remove", [False, True])
async def test_training_start_withdrawal_retains_original_join_observation(native_pg, tmp_path, monkeypatch, surface, remove):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    database = await AsyncDatabase.sqlite(str(tmp_path / "training-local-session.db"))
    await database.execute("CREATE TABLE local_session (id INTEGER PRIMARY KEY, released INTEGER)")
    await database.execute("INSERT INTO local_session VALUES (1, 0)")
    entered, acquire_gate, submit_gate = asyncio.Event(), asyncio.Event(), asyncio.Event()
    withdrawals, accepted = [], []
    async def acquire(*_args):
        entered.set()
        if surface == "opening":
            try:
                await acquire_gate.wait()
            except asyncio.CancelledError:
                pass
        return 1
    async def release(_session):
        await database.execute("UPDATE local_session SET released=released+1 WHERE id=1")
    async def submit(*_args):
        await submit_gate.wait()
        return "declared-local-job"
    lifecycle = SessionTrainingLifecycle(SessionProviderHooks(
        provider_name="declared-local", display_name="Declared local session",
        acquire_session=acquire, session_id=str, submit_job=submit, release_session=release,
    ))
    actual_join = lifecycle_module.await_owned_task
    async def join(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        if outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome
    monkeypatch.setattr(lifecycle_module, "await_owned_task", join)
    class PublicationTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not remove and not self.failed and getattr(callback, "__name__", "") == "mark_complete":
                self.failed = True
                super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
        def remove_done_callback(self, callback):
            count = super().remove_done_callback(callback)
            if remove and not self.failed and getattr(callback, "__name__", "") == "mark_complete":
                self.failed = True
                raise source.error
            return count
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **kwargs):
        code = getattr(_actual_operation(coroutine), "cr_code", None)
        target = SessionTrainingLifecycle._open if surface == "opening" else SessionTrainingLifecycle._run_release
        if code is target.__code__:
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        task = previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
        if surface == "release" and code is SessionTrainingLifecycle._open.__code__:
            starter = asyncio.current_task()
            task.add_done_callback(lambda _done: starter.cancel("actual registered training observer withdrawal"))
        return task
    loop.set_task_factory(factory)
    owner = create_owned_task(lifecycle.start(
        job_id="local", companion_id="declared-companion", trigger_word="local",
        config=TrainingConfig(), avatar_data=b"declared-local-bytes",
    ), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if surface == "opening":
            owner.cancel("actual acquiring training observer withdrawal")
        outcome = await await_owned_task(owner)
        source.assert_retained(outcome.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(outcome.error)) for error in withdrawals)
        assert accepted and all(task.done() for task in accepted)
        assert await database.fetchval("SELECT released FROM local_session WHERE id=1") == 1
        assert not lifecycle._records
    finally:
        loop.set_task_factory(previous)
        acquire_gate.set()
        submit_gate.set()
        await await_owned_task(owner)
        await lifecycle.close()
        await database.close()
