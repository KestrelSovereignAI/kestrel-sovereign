"""Public training adapters preserve actual native receipts, not business text.

Adapters use declared SQLite-backed local session hooks; no cloud provider,
credential, billing, inference or remote job acceptance is exercised here.
"""
import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import is_execution_control_error, execution_cleanup_debts
from kestrel_sovereign.features.training.adapters._session_training_lifecycle import SessionProviderHooks, SessionTrainingLifecycle, ReleaseIntent, SessionJobPhase
from kestrel_sovereign.features.training.adapters import _session_training_lifecycle as lifecycle_module
from kestrel_sovereign.features.training.adapters.runpod_adapter import RunPodTrainingAdapter
from kestrel_sovereign.features.training.adapters.vastai_adapter import VastAITrainingAdapter
from kestrel_sovereign.features.training.adapters.gcp_compute_adapter import GCPComputeTrainingAdapter
from kestrel_sovereign.features.training.types import TrainingConfig
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
async def test_rejected_opening_retains_actual_later_release_observer_withdrawal(native_pg, tmp_path, monkeypatch):
    from kestrel_sovereign.execution_custody import _explicit_control_chain
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    database = await AsyncDatabase.sqlite(str(tmp_path / "rejected-opening-late-withdrawal.db"))
    await database.execute("CREATE TABLE effects (kind TEXT)")
    session, entered, release = object(), asyncio.Event(), asyncio.Event()
    observations, fired = [], []
    async def acquire(*args):
        return session
    async def submit(*args):
        return "declared-job"
    async def close_session(_session):
        assert _session is session
        await database.execute("INSERT INTO effects VALUES ('release')")
        entered.set()
        await release.wait()
    lifecycle = SessionTrainingLifecycle(SessionProviderHooks(provider_name="declared-local",
        display_name="Declared local", acquire_session=acquire, session_id=lambda _: "local",
        submit_job=submit, release_session=close_session))
    actual_join = lifecycle_module.await_owned_task
    async def join(*args, **kwargs):
        result = await actual_join(*args, **kwargs)
        if result.cancellation is not None:
            observations.append(result.cancellation)
        return result
    monkeypatch.setattr(lifecycle_module, "await_owned_task", join)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *args, **kwargs):
            if not fired and getattr(callback, "__self__", None) is lifecycle._openings:
                fired.append(True)
                super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname == "SessionTrainingLifecycle._open":
            return PublicationTask(coroutine, loop=loop, **kwargs)
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    owner = None
    try:
        loop.set_task_factory(factory)
        owner = create_owned_task(lifecycle.start(job_id="original", companion_id="local",
            trigger_word="local", config=TrainingConfig(), avatar_data=b"local"), retain_originals=True)
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual later rejected-opening release observer withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        loop.set_task_factory(previous)
        source.assert_retained(result.error)
        assert fired and observations and all(error in tuple(_explicit_control_chain(result.error)) for error in observations)
        assert not lifecycle.records and not lifecycle._openings
        assert await database.fetchval("SELECT count(*) FROM effects") == 1
    finally:
        loop.set_task_factory(previous)
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        await await_owned_task(create_owned_task(lifecycle.close(), retain_originals=True))
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", [RunPodTrainingAdapter, VastAITrainingAdapter, GCPComputeTrainingAdapter])
@pytest.mark.parametrize("phase", ["acquire", "submission"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_training_public_start_and_status_retain_native_original(
    native_pg, tmp_path, monkeypatch, adapter_type, phase, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    database = await AsyncDatabase.sqlite(str(tmp_path / "public-training-native.db"))
    await database.execute("CREATE TABLE effects (kind TEXT)")
    session = SimpleNamespace(pod_id="declared", instance_id="declared", instance_name="declared", backend_base_url="http://127.0.0.1/unused")
    async def acquire(*args):
        await database.execute("INSERT INTO effects VALUES ('acquire')")
        if phase == "acquire":
            raise source.error
        return session
    async def submit(*args):
        await database.execute("INSERT INTO effects VALUES ('submit')")
        raise source.error
    async def release(_session):
        assert _session is session
        await database.execute("INSERT INTO effects VALUES ('release')")
    lifecycle = SessionTrainingLifecycle(SessionProviderHooks(
        provider_name="declared-local", display_name="Declared local",
        acquire_session=acquire, session_id=lambda _: "declared-local",
        submit_job=submit, release_session=release))
    adapter = object.__new__(adapter_type)
    adapter._lifecycle = lifecycle
    try:
        result = await await_owned_task(create_owned_task(adapter.start_training(
            companion_id="declared", avatar_data=b"local", config=TrainingConfig()), retain_originals=True))
        if phase == "acquire":
            source.assert_retained(result.error)
            assert not lifecycle.records
        else:
            assert result.error is None
            (record,) = lifecycle.records.values()
            original = await await_owned_task(record.submission_task)
            source.assert_retained(original.error)
            status = await await_owned_task(create_owned_task(adapter.get_status(record.job_id), retain_originals=True))
            source.assert_retained(status.error)
            assert is_execution_control_error(status.error)
            closed = await await_owned_task(create_owned_task(lifecycle.close(), retain_originals=True))
            source.assert_retained(closed.error)
            assert record.session_released and record.phase is SessionJobPhase.RELEASED
            assert await database.fetchval("SELECT count(*) FROM effects WHERE kind='release'") == 1
    finally:
        await await_owned_task(create_owned_task(lifecycle.close(), retain_originals=True))
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
async def test_training_release_gate_refusal_keeps_original_task_or_none(native_pg, tmp_path, monkeypatch, accepted):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    database = await AsyncDatabase.sqlite(str(tmp_path / "training-release-gate.db"))
    await database.execute("CREATE TABLE effects (kind TEXT)")
    session, recipes, owners, fired = object(), [], [], []
    async def acquire(*args):
        return session
    async def submit(*args):
        return "declared-local-job"
    async def release(_session):
        assert _session is session
        await database.execute("INSERT INTO effects VALUES ('release')")
    lifecycle = SessionTrainingLifecycle(SessionProviderHooks(provider_name="declared-local",
        display_name="Declared local", acquire_session=acquire, session_id=lambda _: "local",
        submit_job=submit, release_session=release))
    record = await lifecycle.start(job_id="original", companion_id="declared", trigger_word="local", config=TrainingConfig(), avatar_data=b"local")
    await await_owned_task(record.submission_task)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *args, **kwargs):
            if not fired and getattr(callback, "__name__", "") == "close_unstarted":
                fired.append(True)
                super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname == "SessionTrainingLifecycle._run_release":
            recipes.append(operation)
            if not accepted:
                fired.append(True)
                raise source.error
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            owners.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    try:
        loop.set_task_factory(factory)
        result = await await_owned_task(create_owned_task(lifecycle.release("original", ReleaseIntent.CANCEL), retain_originals=True))
        loop.set_task_factory(previous)
        source.assert_retained(result.error)
        assert fired and len(recipes) == 1 and recipes[0].cr_frame is None
        assert record.phase is SessionJobPhase.RELEASE_FAILED and record.native_release_control is not None
        assert record.release_task is (owners[0] if accepted else None)
        assert not owners or owners[0].done()
        assert any(resource is session and task is record.release_task for resource, task in execution_cleanup_debts(result.error))
        repeated = await await_owned_task(create_owned_task(lifecycle.close(), retain_originals=True))
        source.assert_retained(repeated.error)
        assert len(recipes) == 1 and not record.session_released
        assert await database.fetchval("SELECT count(*) FROM effects") == 0
    finally:
        loop.set_task_factory(previous)
        for task in owners:
            await await_owned_task(task)
        # Explicit local fixture reconciliation, never a product retry/ACK.
        await release(session)
        await database.close()
