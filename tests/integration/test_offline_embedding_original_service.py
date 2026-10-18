"""Actual private CLI SDKs retire across resolution and every consumer exit."""
import asyncio

import httpx2 as httpx
import openai
import pytest

from kestrel_sovereign import cli_embeddings
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts, execution_commit_outcome
from kestrel_sovereign.llm import service as service_module
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.setup.constitution_reanchor import _agent_embedding
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_semantic_inference import _GOVERNED_STORAGES, assertion_store as assertion_store
from tests.unit.test_host_context_native_cleanup import _actual_operation


def _private_service(monkeypatch, tmp_path):
    monkeypatch.setattr(service_module, "load_section", lambda _: {})
    service = service_module.LLMService(agent_data_dir=tmp_path / "private-cli")
    http = httpx.AsyncClient(trust_env=False)
    client = openai.AsyncOpenAI(api_key="inert-loopback-only", base_url="http://127.0.0.1:9/v1", http_client=http, max_retries=0)
    service.providers = [{
        "name": "openai:offline", "vendor": "openai", "route": "offline", "model": "offline",
        "client": client, "adapter": OpenAIAdapter(supports_embeddings=False),
        "is_local": True, "is_cloud": False,
    }]
    built = []
    def construct():
        built.append(service)
        return service
    monkeypatch.setattr(service_module, "LLMService", construct)
    return service, http, built


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["resolve", "reindex", "reanchor", "reindex-refusal"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_private_service_resolution_and_cli_exits_retire_actual_sdk(
    native_pg, assertion_store, tmp_path, monkeypatch, door, source_cancel,
):
    original = await _original_control(native_pg[0], monkeypatch, source_cancel)
    service, http, built = _private_service(monkeypatch, tmp_path)
    observed = []
    storage = _GOVERNED_STORAGES[id(assertion_store)]
    db = assertion_store._database
    read = db.fetchall
    async def configuration_read(sql, *args, **kwargs):
        result = await read(sql, *args, **kwargs)
        if "agent_metadata" in sql and not observed:
            observed.append(True)
            raise original
        return result
    if door != "reindex-refusal":
        monkeypatch.setattr(db, "fetchall", configuration_read)
    async def invoke():
        if door == "resolve":
            return await cli_embeddings.resolve_agent_embedding(db, storage.agent_id)
        if door.startswith("reindex"):
            return await cli_embeddings._reindex(
                db, table="saved_items", agent_id=storage.agent_id, batch=1,
                rate_limit=0, dry_run=True, apply=False,
            )
        async with _agent_embedding(db, storage.agent_id, needed=True):
            pytest.fail("failed original configuration entered the reanchor body")
    owner = create_owned_task(invoke(), retain_originals=True)
    try:
        result = await await_owned_task(owner)
        assert built == [service]
        if door == "reindex-refusal":
            assert result.error is None and result.result == 2
        else:
            assert observed and original in tuple(_explicit_control_chain(result.error))
        assert http.is_closed, "private offline embedding service escaped resolution or CLI exit"
        assert not service.has_pending_close_owners()
    finally:
        await await_owned_task(owner)
        await service.close()
        await service.wait_for_close_completion()


@pytest.mark.asyncio
@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
async def test_private_service_canonical_publication_owns_unstarted_sdk(assertion_store, tmp_path, monkeypatch, after, eager):
    from kestrel_sovereign.llm.lifecycle import UnpublishedLLMOwner
    service, http, built = _private_service(monkeypatch, tmp_path)
    storage = _GOVERNED_STORAGES[id(assertion_store)]
    resolution = await cli_embeddings.resolve_agent_embedding(assertion_store._database, storage.agent_id)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refusal, accepted = RuntimeError("original private-service close receipt publication failed"), []
    class OriginalTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *, context=None):
            if self.failed or getattr(callback, "__name__", None) != "close_unstarted":
                return super().add_done_callback(callback, context=context)
            self.failed = True
            if after:
                super().add_done_callback(callback, context=context)
            raise refusal
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is UnpublishedLLMOwner._close.__code__:
            task = OriginalTask(coroutine, loop=loop, **kwargs, **({"eager_start": True} if eager else {}))
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    loop.set_task_factory(factory)
    try:
        for _ in range(2):
            result = await await_owned_task(create_owned_task(resolution.aclose(), retain_originals=True))
            assert refusal in tuple(_explicit_control_chain(result.error))
            assert len(accepted) == 1 and accepted[0].done(), "private service lost its original accepted closer"
            assert any(resource is service and task is accepted[0] for resource, task in execution_cleanup_debts(result.error))
            assert not http.is_closed, "unpublished private service closer spent native retirement"
        assert built == [service] and not resolution._private_owner.retired
    finally:
        loop.set_task_factory(previous)
        for task in accepted:
            await await_owned_task(task)
        await service.close()
        await service.wait_for_close_completion()


@pytest.mark.asyncio
async def test_retired_private_service_never_admits_another_close_task(assertion_store, tmp_path, monkeypatch):
    from kestrel_sovereign.llm.lifecycle import UnpublishedLLMOwner
    service, http, built = _private_service(monkeypatch, tmp_path)
    storage = _GOVERNED_STORAGES[id(assertion_store)]
    resolution = await cli_embeddings.resolve_agent_embedding(assertion_store._database, storage.agent_id)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    admitted = []
    def factory(loop, coroutine, **kwargs):
        task = previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
        if getattr(_actual_operation(coroutine), "cr_code", None) is UnpublishedLLMOwner._close.__code__:
            admitted.append(task)
        return task
    loop.set_task_factory(factory)
    try:
        await resolution.aclose()
        await resolution.aclose()
        assert built == [service] and http.is_closed and resolution._private_owner.retired
        assert len(admitted) == 1 and admitted[0].done(), "retired private service admitted a replacement close owner"
    finally:
        loop.set_task_factory(previous)
        await service.close()
        await service.wait_for_close_completion()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["resolution", "service", "remote"])
async def test_retired_private_owner_still_replays_actual_nonlocal_observer_control(native_pg, assertion_store, tmp_path, monkeypatch, surface):
    from kestrel_sovereign.llm import lifecycle, remote_backend
    from tests.utils.original_failure import OriginalFailure
    original = await _original_control(native_pg[0], monkeypatch)
    source = OriginalFailure.capture(original)
    service, http, _ = _private_service(monkeypatch, tmp_path)
    storage = _GOVERNED_STORAGES[id(assertion_store)]
    resolution = await cli_embeddings.resolve_agent_embedding(assertion_store._database, storage.agent_id) if surface == "resolution" else None
    entered, release = asyncio.Event(), asyncio.Event()
    module = {"resolution": lifecycle, "service": service_module, "remote": remote_backend}[surface]
    actual_close, actual_join = httpx.AsyncClient.aclose, module.await_owned_task
    closes, withdrawals = [], []
    async def close(self):
        await actual_close(self)
        if self is http:
            closes.append(asyncio.current_task())
            entered.set()
            await release.wait()
    async def join(task, *args, **kwargs):
        result = await actual_join(task, *args, **kwargs)
        if result.cancellation is not None:
            # A real join observer can carry earlier native uncertainty.
            # This does not change the original successful physical closer.
            result.cancellation.__cause__ = original
            withdrawals.append(result.cancellation)
        return result
    monkeypatch.setattr(httpx.AsyncClient, "aclose", close)
    monkeypatch.setattr(module, "await_owned_task", join)
    client = service.providers[0]["client"]
    if surface == "service":
        lifecycle.start_llm_close_task(service, ("client", id(client)), client.close, resource=client)
    async def retire():
        if surface == "resolution":
            await resolution.aclose()
        elif surface == "service":
            await service.wait_for_close_completion()
        else:
            await service._close_remote_client(client, lease_id="declared-local-native-replay")
    owner = create_owned_task(retire(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual private-service observer with prior native control")
        await asyncio.sleep(0)
        release.set()
        first = await await_owned_task(owner)
        repeated = await await_owned_task(create_owned_task(retire(), retain_originals=True))
        for result in (first, repeated):
            source.assert_retained(result.error)
            assert execution_commit_outcome(result.error) == "unknown"
        assert withdrawals and all(error in tuple(_explicit_control_chain(first.error)) for error in withdrawals)
        assert all(error not in tuple(_explicit_control_chain(repeated.error)) for error in withdrawals), "native source retention stored the caller wrapper as shared history"
        assert http.is_closed and not service.has_pending_close_owners()
        if resolution is not None:
            assert resolution._private_owner.retired
        assert len(closes) == 1 and closes[0].done()
    finally:
        release.set()
        await await_owned_task(owner)
        for cleanup in (service.close, service.wait_for_close_completion):
            settled = await await_owned_task(create_owned_task(cleanup(), retain_originals=True))
            if settled.error is not None:
                source.assert_retained(settled.error)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("withdraw", [False, True])
async def test_reanchor_retirement_keeps_body_close_and_observer_originals(native_pg, assertion_store, tmp_path, monkeypatch, source_cancel, withdraw):
    from kestrel_sovereign import _async_ownership as ownership
    from kestrel_sovereign.llm import lifecycle
    service, http, built = _private_service(monkeypatch, tmp_path)
    storage = _GOVERNED_STORAGES[id(assertion_store)]
    actual_close, actual_join = httpx.AsyncClient.aclose, ownership.await_owned_task
    entered, release = asyncio.Event(), asyncio.Event()
    close_error = OSError("actual private SDK pool close observer failed")
    bodies, closers, withdrawals = [], [], []
    async def close(self):
        await actual_close(self)
        if self is http:
            closers.append(asyncio.current_task())
            entered.set()
            await release.wait()
            raise close_error
    async def join(task, *args, **kwargs):
        result = await actual_join(task, *args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    monkeypatch.setattr(httpx.AsyncClient, "aclose", close)
    monkeypatch.setattr(ownership, "await_owned_task", join)
    monkeypatch.setattr(lifecycle, "await_owned_task", join)
    async def invoke():
        async with _agent_embedding(assertion_store._database, storage.agent_id, needed=True):
            original = await _original_control(native_pg[0], monkeypatch, source_cancel)
            bodies.append((original, tuple(_explicit_control_chain(original)), tuple(execution_cleanup_debts(original))))
            raise original
    owner = create_owned_task(invoke(), retain_originals=True)
    waiter = create_owned_task(entered.wait())
    try:
        await asyncio.wait({owner, waiter}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
        assert entered.is_set(), "reanchor body escaped original private service retirement admission"
        if withdraw:
            owner.cancel("actual reanchor cleanup observer withdrew")
            await asyncio.sleep(0)
            assert not owner.done()
        release.set()
        result = await await_owned_task(owner)
        original, graph, debts = bodies[0]
        chain = tuple(_explicit_control_chain(result.error))
        assert original in chain and close_error in chain
        assert execution_commit_outcome(result.error) == "unknown"
        if withdraw:
            assert withdrawals and all(error in chain for error in withdrawals), "reanchor dropped actual independent cleanup withdrawal"
        assert tuple(_explicit_control_chain(original)) == graph, "reanchor rewrote original body control graph"
        assert tuple(execution_cleanup_debts(original)) == debts, "reanchor appended private retirement debt to shared body control"
        assert http.is_closed and len(closers) == 1 and closers[0].done() and built == [service]
        assert any(resource is service and task is not None and task.done() for resource, task in execution_cleanup_debts(result.error))
    finally:
        release.set()
        if not waiter.done():
            waiter.cancel()
        await await_owned_task(waiter)
        await await_owned_task(owner)
        for operation in (service.close, service.wait_for_close_completion):
            settled = await await_owned_task(create_owned_task(operation(), retain_originals=True))
            if settled.error is not None and close_error not in tuple(_explicit_control_chain(settled.error)):
                raise settled.error
