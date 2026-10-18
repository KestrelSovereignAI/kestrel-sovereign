"""Real native consumers retain join evidence separately from physical results.

Only disposable PostgreSQL/SQLite and in-process SDK pools are used. No
provider request, model inference, authenticated HTTP or genesis/READY claim.
"""
import asyncio
from contextvars import ContextVar

import asyncpg
import httpx
import openai
import pytest
from google import genai

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.features.health.feature import HealthFeature
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.llm import genai_custody
from kestrel_sovereign.native_retirement import NativeExecutionObserver, bind_native_execution_observer
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["service-wait", "unpublished", "remote-close"])
@pytest.mark.parametrize("remove", [False, True])
@pytest.mark.parametrize("withdraw", [False, True])
async def test_native_service_observation_keeps_withdrawal_out_of_shared_control(native_pg, tmp_path, monkeypatch, surface, remove, withdraw):
    import httpx2 as httpx
    from kestrel_sovereign.llm import lifecycle as lifecycle_module, service as service_module, remote_backend as remote_module
    from kestrel_sovereign.llm.lifecycle import UnpublishedLLMOwner, retire_unpublished_llm, start_llm_close_task

    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    monkeypatch.setattr(service_module, "load_section", lambda _: {})
    service = service_module.LLMService(agent_data_dir=tmp_path / "original-native-service")
    client = openai.AsyncOpenAI(api_key="inert-no-provider-request", base_url="http://127.0.0.1:1", max_retries=0)
    actual_close = httpx.AsyncClient.aclose
    entered, release, closes, withdrawals = asyncio.Event(), asyncio.Event(), [], []
    async def close(native):
        await actual_close(native)
        if native is client._client:
            closes.append(asyncio.current_task())
            entered.set()
            await release.wait()
    monkeypatch.setattr(httpx.AsyncClient, "aclose", close)
    module = {"service-wait": service_module, "unpublished": lifecycle_module, "remote-close": remote_module}[surface]
    actual_join = module.await_owned_task
    async def join(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        if outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome
    monkeypatch.setattr(module, "await_owned_task", join)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    task_type = _publication_task_type(source.error, remove=remove, after=True)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code, frame = getattr(operation, "cr_code", None), getattr(operation, "cr_frame", None)
        selected = (code is UnpublishedLLMOwner._close.__code__) if surface == "unpublished" else (
            code is not None and code.co_name == "close" and frame.f_locals.get("self") is client)
        if selected:
            return task_type(coroutine, loop=loop, **kwargs)
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    owners = {}
    async def retire():
        if surface == "unpublished":
            await retire_unpublished_llm(owners, "original-service", service, (), join_native=True)
        elif surface == "remote-close":
            await service._close_remote_client(client, lease_id="declared-local-route")
        else:
            await service.wait_for_close_completion()
    loop.set_task_factory(factory)
    owner = None
    try:
        if surface == "unpublished":
            service.providers = [{"name": "no-request-local-native", "client": client}]
        elif surface == "service-wait":
            start_llm_close_task(service, "observed-native-client", client.close, resource=client)
        owner = create_owned_task(retire(), retain_originals=True)
        await asyncio.wait_for(entered.wait(), 5)
        if withdraw:
            owner.cancel("actual native service retirement observer withdrew")
            await asyncio.sleep(0)
            assert not owner.done()
        release.set()
        first = await await_owned_task(owner)
        source.assert_retained(first.error)
        loop.set_task_factory(previous)
        repeated = await await_owned_task(create_owned_task(retire(), retain_originals=True))
        source.assert_retained(repeated.error)
        assert client._client.is_closed and len(closes) == 1 and closes[0].done()
        assert not execution_cleanup_debts(first.error), "successful physical service closure invented native resource debt"
        if withdraw:
            assert withdrawals and all(error in tuple(_explicit_control_chain(first.error)) for error in withdrawals)
            assert all(error not in tuple(_explicit_control_chain(repeated.error)) for error in withdrawals), "finite service observer withdrawal leaked into shared control history"
    finally:
        loop.set_task_factory(previous)
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        await actual_close(client._client)
        # Join the same service, including its latched original evidence.
        # Remove no native resource or pending Task to suppress that evidence.
        for cleanup in (service.close, service.wait_for_close_completion):
            await await_owned_task(create_owned_task(cleanup(), retain_originals=True))


def _publication_task_type(original, *, remove, after, enabled=lambda: True):
    class PublicationTask(asyncio.Task):
        refused = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not remove and not self.refused and enabled() and getattr(callback, "__name__", "") == "mark_complete":
                self.refused = True
                if after:
                    super().add_done_callback(callback, *args, **kwargs)
                raise original
            return super().add_done_callback(callback, *args, **kwargs)
        def remove_done_callback(self, callback):
            if remove and not self.refused and enabled() and getattr(callback, "__name__", "") == "mark_complete":
                self.refused = True
                if after:
                    super().remove_done_callback(callback)
                raise original
            return super().remove_done_callback(callback)
    return PublicationTask


@pytest.mark.asyncio
@pytest.mark.parametrize("remove", [False, True])
@pytest.mark.parametrize("after", [False, True])
async def test_health_shutdown_keeps_observation_of_original_pre_ready_resident(native_pg, onboarding_owner, monkeypatch, remove, after):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    _manager, agent, database = onboarding_owner
    feature = HealthFeature(agent)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    task_type = _publication_task_type(source.error, remove=remove, after=after)
    accepted = []
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        frame = getattr(operation, "cr_frame", None)
        resident = frame.f_locals.get("coro") if frame is not None else None
        code = getattr(resident, "cr_code", None)
        if code is not None and code.co_qualname == "HealthFeature._start_background_loop.<locals>.run":
            task = task_type(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    loop.set_task_factory(factory)
    try:
        await feature.initialize()
        assert len(accepted) == 1 and not agent._runtime_publication_ready.is_set()
        first = await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))
        source.assert_retained(first.error)
        assert accepted[0].done() and feature._background_task is None
        again = await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))
        source.assert_retained(again.error)
        assert await database.table_exists("health_log")
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))
        for task in accepted:
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("remove", [False, True])
@pytest.mark.parametrize("surface", ["next", "close"])
async def test_iterator_terminal_projection_keeps_independent_join_control(native_pg, onboarding_owner, monkeypatch, remove, surface):
    from kestrel_sovereign._async_ownership import OwnedAsyncIterator
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    physical = OriginalFailure.capture(OSError("actual iterator source failed after native write"))
    _manager, _agent, database = onboarding_owner
    async def produce():
        await database.execute("INSERT INTO host_rollback_effects VALUES (12)")
        try:
            if surface == "close":
                yield await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=12")
            raise physical.error
        finally:
            if surface == "close":
                raise physical.error
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    task_type = _publication_task_type(source.error, remove=remove, after=True,
        enabled=lambda: not asyncio.current_task().get_name().startswith("owned_async_iterator_cancel:"))
    accepted = []
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname == "OwnedAsyncIterator.__init__.<locals>.run_owned":
            task = task_type(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    loop.set_task_factory(factory)
    stream = OwnedAsyncIterator(produce, operation="native-iterator-projection")
    try:
        if surface == "close":
            assert await anext(stream) == 1
        observed = await await_owned_task(create_owned_task(
            stream.aclose() if surface == "close" else anext(stream), retain_originals=True,
        ))
        source.assert_retained(observed.error)
        physical.assert_retained(observed.error)
        assert accepted and all(task.done() for task in accepted)
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=12") == 1
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(create_owned_task(stream.aclose(), retain_originals=True))


@pytest.mark.asyncio
@pytest.mark.parametrize("remove", [False, True])
@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("withdraw", [False, True])
async def test_genai_successful_close_keeps_join_control_without_phantom_debt(native_pg, monkeypatch, remove, after, withdraw):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    client = genai.Client(api_key="inert-no-provider-request")
    genai_custody.prepare_genai_client(client)
    http = client._kestrel_genai_httpx_owned
    actual_close, actual_join = httpx.AsyncClient.aclose, genai_custody.await_owned_task
    entered, release, closers, withdrawals = asyncio.Event(), asyncio.Event(), [], []
    async def close(native):
        await actual_close(native)
        if native is http:
            closers.append(asyncio.current_task())
            entered.set()
            await release.wait()
    async def join(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        if outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome
    monkeypatch.setattr(httpx.AsyncClient, "aclose", close)
    monkeypatch.setattr(genai_custody, "await_owned_task", join)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    task_type = _publication_task_type(source.error, remove=remove, after=after)
    def factory(loop, coroutine, **kwargs):
        if getattr(_actual_operation(coroutine), "cr_code", None) is genai_custody._close_genai_layers.__code__:
            return task_type(coroutine, loop=loop, **kwargs)
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    owner = create_owned_task(genai_custody.close_genai_client(client), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if withdraw:
            owner.cancel("actual GenAI join observer withdrew")
            await asyncio.sleep(0)
            assert not owner.done()
        release.set()
        outcome = await await_owned_task(owner)
        source.assert_retained(outcome.error)
        assert len(closers) == 1 and closers[0].done() and http.is_closed
        assert not execution_cleanup_debts(outcome.error), "successful GenAI close invented native debt"
        assert client._llm_close_tasks["genai-native-client"] is None
        loop.set_task_factory(previous)
        again = await await_owned_task(create_owned_task(genai_custody.close_genai_client(client), retain_originals=True))
        source.assert_retained(again.error)
        assert len(closers) == 1
        if withdraw:
            assert withdrawals and all(error in tuple(_explicit_control_chain(outcome.error)) for error in withdrawals)
            assert all(error not in tuple(_explicit_control_chain(again.error)) for error in withdrawals)
    finally:
        loop.set_task_factory(previous)
        release.set()
        await await_owned_task(owner)
        await actual_close(http)
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("remove", [False, True])
@pytest.mark.parametrize("after", [False, True])
async def test_native_inventory_keeps_join_error_after_successful_physical_commit(native_pg, monkeypatch, remove, after):
    backend, control, schema = native_pg
    source = OriginalFailure.capture(await _original_control(backend, monkeypatch))
    inventory = NativeExecutionObserver()
    entered, release = asyncio.Event(), asyncio.Event()
    observing = ContextVar("fault_only_actual_inventory_observer", default=False)
    native_exit = asyncpg.transaction.Transaction.__aexit__
    async def exit(transaction, *args):
        result = await native_exit(transaction, *args)
        entered.set()
        await release.wait()
        return result
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", exit)
    task_type = _publication_task_type(source.error, remove=remove, after=after, enabled=observing.get)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    accepted = []
    def factory(loop, coroutine, **kwargs):
        if getattr(_actual_operation(coroutine), "cr_code", None) is exit.__code__:
            task = task_type(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    async def work():
        with bind_native_execution_observer(inventory):
            async with backend.transaction():
                await backend.execute("INSERT INTO effects VALUES (245, 'actual successful native observer commit')")
    async def observe():
        token = observing.set(True)
        try:
            return await inventory.join_accepted()
        finally:
            observing.reset(token)
    owner, joined = None, None
    loop.set_task_factory(factory)
    try:
        owner = create_owned_task(work(), retain_originals=True)
        await asyncio.wait_for(entered.wait(), 5)
        joined = create_owned_task(observe(), retain_originals=True)
        await asyncio.sleep(0)
        release.set()
        observed = await await_owned_task(joined)
        assert observed.error is None
        source.assert_retained(observed.result.error)
        physical = await await_owned_task(owner)
        assert physical.error is None and accepted and all(task.done() for task in accepted)
        assert not inventory._resources, "successful native reset retained phantom physical ownership"
        source.assert_retained((await inventory.join_accepted()).error)
        assert await control.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=245') == 1
    finally:
        loop.set_task_factory(previous)
        release.set()
        if joined is not None:
            await await_owned_task(joined)
        if owner is not None:
            await await_owned_task(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["feature", "scheduler"])
@pytest.mark.parametrize("remove", [False, True])
@pytest.mark.parametrize("after", [False, True])
async def test_real_resident_owner_keeps_join_control_after_worker_success(native_pg, onboarding_owner, monkeypatch, surface, remove, after):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    _manager, agent, database = onboarding_owner
    entered, release, tasks = asyncio.Event(), asyncio.Event(), []
    class ResidentFeature(Feature):
        @property
        def tool_description(self):
            return "Actual resident cleanup probe"
        async def initialize(self):
            return None
    feature = ResidentFeature(agent)
    runner = SchedulerRunner(AsyncDatabase(native_pg[0]), agent.did, lambda *args: None)
    async def work():
        try:
            entered.set()
            await release.wait()
        finally:
            await database.execute("INSERT INTO host_rollback_effects VALUES (11)")
    task_type = _publication_task_type(source.error, remove=remove, after=after)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **kwargs):
        if getattr(_actual_operation(coroutine), "cr_code", None) is work.__code__:
            task = task_type(coroutine, loop=loop, **kwargs)
            tasks.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    async def retire():
        if surface == "feature":
            await feature._cancel_owned_background_tasks()
        else:
            await runner._join_runtime_status_publish_task(deadline=loop.time()+1)
            if runner.readiness_failure is not None:
                raise runner.readiness_failure
    loop.set_task_factory(factory)
    owner = None
    try:
        if surface == "feature":
            feature._track_owned_background_task(work(), name="actual-feature-resident")
        else:
            runner._runtime_status_publish_task = create_owned_task(work(), retain_originals=True)
        await asyncio.wait_for(entered.wait(), 5)
        # The feature owns cooperative interruption; make its actual worker
        # finish normally so only observation, not the Task result, carries P1.
        owner = create_owned_task(retire(), retain_originals=True)
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert len(tasks) == 1 and tasks[0].done()
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=11") == 1
        assert (not feature._owned_background_tasks) if surface == "feature" else runner._runtime_status_publish_task is None
    finally:
        loop.set_task_factory(previous)
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        for task in tasks:
            await await_owned_task(task)
