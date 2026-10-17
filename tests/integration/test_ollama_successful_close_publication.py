"""Successful real pool close cannot erase independent lifecycle control."""
import asyncio

import httpx
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
from kestrel_sovereign.llm import ollama_adapter as adapter_module, service as service_module
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.integration.test_ollama_discovery_original_retirement import catalog_peer as catalog_peer
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("embedding", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("withdraw", [False, True])
@pytest.mark.parametrize("callback_stage", ["generic", "join-add", "join-remove"])
async def test_successful_pool_retirement_replays_publication_not_local_withdrawal(
    native_pg, catalog_peer, monkeypatch, tmp_path, after, embedding, source_cancel, withdraw, callback_stage,
):
    clients, requests, _control, actual_close = catalog_peer
    original = await _original_control(native_pg[0], monkeypatch, source_cancel)
    source = OriginalFailure.capture(original)
    entered, release = asyncio.Event(), asyncio.Event()
    closers, withdrawals = [], []
    adapter = OllamaAdapter()
    monkeypatch.setattr(service_module, "load_section", lambda _: {})
    service = service_module.LLMService(agent_data_dir=tmp_path / "ollama-publication")
    service.providers = [{"name": "local-ollama", "adapter": adapter}]
    async def close(native):
        await actual_close(native)
        if clients and native is clients[0]._client:
            closers.append(asyncio.current_task())
            entered.set()
            await release.wait()
    actual_join = adapter_module.await_owned_task
    async def join(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        if outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome
    monkeypatch.setattr(httpx.AsyncClient, "aclose", close)
    monkeypatch.setattr(adapter_module, "await_owned_task", join)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    class PublicationTask(asyncio.Task):
        refused = False
        def add_done_callback(self, callback, *args, **kwargs):
            selected = "completed" if callback_stage == "generic" else "mark_complete"
            if callback_stage != "join-remove" and not self.refused and getattr(callback, "__name__", "") == selected:
                self.refused = True
                if after:
                    super().add_done_callback(callback, *args, **kwargs)
                raise original
            return super().add_done_callback(callback, *args, **kwargs)
        def remove_done_callback(self, callback):
            if callback_stage == "join-remove" and not self.refused and getattr(callback, "__name__", "") == "mark_complete":
                self.refused = True
                if after:
                    super().remove_done_callback(callback)
                raise original
            return super().remove_done_callback(callback)
    def factory(loop, coroutine, **kwargs):
        if getattr(_actual_operation(coroutine), "cr_code", None) is close.__code__:
            return PublicationTask(coroutine, loop=loop, **kwargs)
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    invoke = adapter.list_embedding_models if embedding else adapter.list_models
    owner = create_owned_task(invoke(), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if withdraw:
            owner.cancel("actual successful pool observer withdrew")
            await asyncio.sleep(0)
            assert not owner.done()
        release.set()
        outcome = await await_owned_task(owner)
        source.assert_retained(outcome.error)
        assert len(closers) == 1 and closers[0].done() and clients[0]._client.is_closed
        assert not adapter._discovery_clients, "successful pool retained obsolete resource ownership"
        assert not execution_cleanup_debts(outcome.error), "successful native close invented unresolved pool debt"
        if withdraw:
            assert withdrawals and all(error in tuple(_explicit_control_chain(outcome.error)) for error in withdrawals)
        fresh = await await_owned_task(create_owned_task(invoke(), retain_originals=True))
        source.assert_retained(fresh.error)
        assert len(clients) == len(requests) == 1, "retained publication control admitted another private discovery pool"
        loop.set_task_factory(previous)
        for retire in (adapter.aclose, adapter.aclose, service.close, service.wait_for_close_completion):
            repeated = await await_owned_task(create_owned_task(retire(), retain_originals=True))
            source.assert_retained(repeated.error)
            assert all(error not in tuple(_explicit_control_chain(repeated.error)) for error in withdrawals), "finite observer withdrawal leaked into shared adapter history"
        assert len(closers) == 1 and len(requests) == 1
    finally:
        loop.set_task_factory(previous)
        release.set()
        await await_owned_task(owner)
        for retire in (service.close, service.wait_for_close_completion):
            await await_owned_task(create_owned_task(retire(), retain_originals=True))
