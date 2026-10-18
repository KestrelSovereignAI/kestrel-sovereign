"""Real-consumer regressions for custody review9 (Core #3569).

No providers or real child processes are contacted. Gates use event-controlled
adapter/lifecycle boundaries, exercise production consumers, and join every
task/worker on both passing and deliberately broken source trees.
"""

import asyncio
from contextlib import asynccontextmanager
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    ExecutionCustody,
    execution_commit_outcome,
)
from kestrel_sovereign.llm.service import LLMService
from tests.unit.test_execution_custody import Authority
from tests.unit.test_decisions_service import (
    FakeDecisionAdapter, _request, _route, _service,
)
from tests.unit.test_isolated_feature_runtime import (
    FakeIsolatedClient, _initialized_host_ingress_proxy,
)
from tests.unit.test_streaming_usage_metering import _FakeService


def _unknown(error_type=RuntimeError):
    error = error_type("native storage wrapper")
    error.__cause__ = ExecutionCommitOutcomeError("unknown")
    return error


@pytest.mark.asyncio
@pytest.mark.parametrize("local_only", [False, True])
async def test_lazy_discovery_control_never_continues_routing(local_only):
    service = LLMService.__new__(LLMService)
    service._available_providers = lambda: [{"model": "auto", "is_local": True}]
    error = _unknown()
    service.discover_all_models = AsyncMock(side_effect=error)
    service._resolve_local_auto_routes = AsyncMock(side_effect=error)
    service.resolve_provider_routing = Mock(return_value=object())
    with pytest.raises(RuntimeError) as caught:
        await service._resolve_routing_with_discovery(force_local_only=local_only)
    assert caught.value is error
    service.resolve_provider_routing.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["track", "record"])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_actual_usage_writer_preserves_transaction_outcome(monkeypatch, consumer, wrapped):
    from kestrel_sovereign.llm import usage_tracking
    service = LLMService.__new__(LLMService)
    error = _unknown() if wrapped else ExecutionCommitOutcomeError("unknown")

    @asynccontextmanager
    async def transaction():
        yield
        raise error

    service._usage_db = SimpleNamespace(transaction=transaction, execute=AsyncMock())
    service._db_initialized = True
    classify = Mock(return_value=None)
    monkeypatch.setattr(usage_tracking, "concurrent_write_retry_delay", classify)
    with pytest.raises(RuntimeError) as caught:
        if consumer == "track":
            await service._track_model_usage("model", "route", tokens=5)
        else:
            await service._record_model_usage("model", "route", tokens=5, label="chat")
    assert caught.value is error
    assert execution_commit_outcome(caught.value) == "unknown"
    assert service._usage_db.execute.await_count == 2
    classify.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("body_error", [ValueError, asyncio.CancelledError])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_aborted_stream_accounting_retains_irreversible_outcome(body_error, wrapped):
    service = _FakeService()
    error = _unknown() if wrapped else ExecutionCommitOutcomeError("unknown")
    service._record_streamed_usage = AsyncMock(side_effect=error)

    class Adapter:
        supports_partial_usage_flush = True

        async def get_streaming_response_with_tools(self, *, usage_sink, **kwargs):
            usage_sink["input_tokens"] = 23
            yield "partial"
            raise body_error("provider aborted")

    stream = service._stream_adapter_with_usage(
        adapter=Adapter(), client=None, model="model", messages=[],
        provider_name="synthetic", path="stream", invocation_context=None,
        expose_protocol_events=False,
    )
    seen = []
    try:
        with pytest.raises(RuntimeError) as caught:
            async for item in stream:
                seen.append(item)
        assert caught.value is error
        assert execution_commit_outcome(caught.value) == "unknown"
        assert seen == ["partial"]
        service._record_streamed_usage.assert_awaited_once()
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["chat", "embedding"])
@pytest.mark.parametrize("loss", ["before", "during"])
async def test_discovery_actual_dispatch_and_publication_are_fenced(consumer, loss):
    service = LLMService.__new__(LLMService)
    scope = ExecutionCustody(Authority())
    service._execution_custody = scope
    service._note_discovery_outcome = Mock()
    model = SimpleNamespace(id="synthetic", provider="original", route="original")

    async def listing(*args, **kwargs):
        scope.revoke("lost original generation")
        return [model] if consumer == "embedding" else []

    listing_mock = AsyncMock(side_effect=listing)
    adapter = SimpleNamespace(list_models=listing_mock, list_embedding_models=listing_mock)
    if loss == "before":
        scope.revoke("lost original generation")
    with pytest.raises(ExecutionAuthorityError, match="lost original generation"):
        if consumer == "chat":
            await service._safe_list_models("synthetic", adapter, None)
        else:
            await service._discover_embedding_for_route(
                "synthetic", {"name": "synthetic:route", "adapter": adapter},
            )
    if loss == "before":
        listing_mock.assert_not_awaited()
    else:
        listing_mock.assert_awaited_once()
    service._note_discovery_outcome.assert_not_called()
    assert (model.provider, model.route) == ("original", "original")


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["chat", "embedding"])
async def test_discovery_tolerance_preserves_cause_chained_control(consumer):
    service = LLMService.__new__(LLMService)
    service._note_discovery_outcome = Mock()
    error = _unknown()
    listing = AsyncMock(side_effect=error)
    adapter = SimpleNamespace(list_models=listing, list_embedding_models=listing)
    with pytest.raises(RuntimeError) as caught:
        if consumer == "chat":
            await service._safe_list_models("synthetic", adapter, None)
        else:
            await service._discover_embedding_for_route("synthetic", {"adapter": adapter})
    assert caught.value is error
    service._note_discovery_outcome.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, "native_query"])
async def test_canary_unlisted_control_never_records_success(error_type):
    from kestrel_sdk.storage.database import QueryError
    error = _unknown(QueryError if error_type == "native_query" else error_type)
    adapter = FakeDecisionAdapter([], error=error)
    provider = _route("ollama:local", adapter, local=True, pin="synthetic")
    service = _service([provider])
    with pytest.raises(type(error)) as caught:
        await service._run_pin_canary(provider, provider["decision_state"])
    assert caught.value is error
    service._track_model_usage.assert_not_awaited()
    service._observability_store.log_llm_call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["reconcile", "decide"])
async def test_decision_fanout_joins_sibling_and_selects_late_outcome(consumer):
    entered = asyncio.Event()
    settled = asyncio.Event()
    children = []
    late = ExecutionCommitOutcomeError("unknown")

    class Denied(FakeDecisionAdapter):
        async def list_decision_models(self, client):
            await entered.wait()
            raise ExecutionAuthorityError("discovery denied")

    class Sibling(FakeDecisionAdapter):
        async def list_decision_models(self, client):
            children.append(asyncio.current_task())
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0)
                raise late
            finally:
                settled.set()

    denied, sibling = Denied([]), Sibling([])
    service = _service([
        _route("ollama:first", denied, local=True),
        _route("ollama:second", sibling, local=True),
    ])
    try:
        with pytest.raises(ExecutionAuthorityError) as caught:
            if consumer == "reconcile":
                await service.reconcile_decision_capabilities()
            else:
                await service.decide(_request(), caller="synthetic", timeout_seconds=5)
        assert settled.is_set(), "fanout returned before its sibling was cancelled/joined"
        assert all(child.done() for child in children)
        assert caught.value is late
        assert denied.decide_calls == sibling.decide_calls == []
        service._track_model_usage.assert_not_awaited()
    finally:
        for child in children:
            if not child.done():
                child.cancel()
        await asyncio.gather(*children, return_exceptions=True)


@pytest.mark.asyncio
async def test_idle_wake_rechecks_original_scope_after_lifecycle_lock(monkeypatch, tmp_path):
    feature, agent = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, FakeIsolatedClient,
    )
    client = feature._client
    feature._client = None
    feature._idle_retired = True
    scope = ExecutionCustody(Authority())
    agent._execution_custody = scope
    prepare = Mock(wraps=feature._prepare_runtime_workspace)
    monkeypatch.setattr(feature, "_prepare_runtime_workspace", prepare)
    entered = asyncio.Event()

    async def wake():
        entered.set()
        await feature._wake_idle_runtime_uninterrupted()

    task = None
    try:
        await feature._reload_lock.acquire()
        task = asyncio.create_task(wake())
        await asyncio.wait_for(entered.wait(), timeout=1)
        scope.revoke("lost original generation")
        feature._reload_lock.release()
        with pytest.raises(ExecutionAuthorityError, match="lost original generation"):
            await task
        prepare.assert_not_called()
        assert feature._client is None
    finally:
        if feature._reload_lock.locked():
            feature._reload_lock.release()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        feature._client = client
        await feature.shutdown()


@pytest.mark.asyncio
async def test_venv_worker_rechecks_scope_before_synchronous_mutation(monkeypatch, tmp_path):
    feature, agent = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, FakeIsolatedClient,
    )
    scope = ExecutionCustody(Authority())
    agent._execution_custody = scope
    entered, release = Event(), Event()
    original = feature.ensure_venv
    mutation = Mock(return_value=False)
    monkeypatch.setattr(feature, "_ensure_venv_with_active_budget", mutation)

    def delayed_worker():
        entered.set()
        assert release.wait(timeout=5), "test did not release owned worker"
        return original()

    monkeypatch.setattr(feature, "ensure_venv", delayed_worker)
    task = asyncio.create_task(feature._ensure_venv_without_blocking_event_loop())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        scope.revoke("lost original generation")
        release.set()
        with pytest.raises(ExecutionAuthorityError, match="lost original generation"):
            await task
        mutation.assert_not_called()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await feature.shutdown()


@pytest.mark.asyncio
async def test_supervisor_restart_predicate_denies_retired_original_scope(monkeypatch, tmp_path):
    feature, agent = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, FakeIsolatedClient,
    )
    scope = ExecutionCustody(Authority())
    agent._execution_custody = scope
    scope.revoke("lost original generation")
    try:
        with pytest.raises(ExecutionAuthorityError, match="lost original generation"):
            feature._supervisor_owns_client_restart(feature._client, feature._reload_gen)
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["tool", "channel"])
async def test_isolated_rpc_denies_result_publication_after_authority_loss(monkeypatch, tmp_path, consumer):
    from kestrel_sovereign.features.isolated_runtime import ProxyChannelAdapter
    feature, agent = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, FakeIsolatedClient,
    )
    scope = ExecutionCustody(Authority())
    agent._execution_custody = scope

    async def call(*args):
        scope.revoke("lost original generation")
        return {"status": "success", "data": {"message_id": "synthetic"}}

    feature._client.call_tool = AsyncMock(side_effect=call)
    publish = Mock()
    monkeypatch.setattr(feature, "_maybe_emit_channel_link_part", publish)
    try:
        with pytest.raises(ExecutionAuthorityError, match="lost original generation"):
            if consumer == "tool":
                await feature.call_isolated_tool("ping", {"message": "synthetic"})
            else:
                adapter = ProxyChannelAdapter(feature, channel_type="synthetic", send_tool="ping")
                await adapter.send_message("synthetic", "message")
        feature._client.call_tool.assert_awaited_once()
        publish.assert_not_called()
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("loss", ["backoff", "health", "restart"])
async def test_actual_supervisor_fences_loss_and_joins_child(monkeypatch, tmp_path, loss):
    scope = ExecutionCustody(Authority())
    health_called = asyncio.Event()

    class ControlledClient(FakeIsolatedClient):
        starts = 0
        stops = 0

        async def start(self):
            self.starts += 1
            await super().start()
            if self.starts > 1:
                scope.revoke("lost original generation")

        async def health(self):
            health_called.set()
            if loss == "health":
                scope.revoke("lost original generation")
            return False

        async def stop(self):
            self.stops += 1
            await super().stop()

    client = ControlledClient()
    feature, agent = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **kwargs: client,
    )
    agent._execution_custody = scope
    task = feature._supervision_task
    try:
        if loss == "backoff":
            scope.revoke("lost original generation")
        with pytest.raises(ExecutionAuthorityError, match="lost original generation"):
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        assert task.done()
        assert client.stopped
        assert feature._client is None
        assert feature._traffic_gate.sealed
        if loss == "backoff":
            assert not health_called.is_set()
            assert client.starts == 1
        elif loss == "health":
            assert health_called.is_set()
            assert client.starts == 1
        else:
            assert client.starts == 2
            assert client.stops == 2
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await feature.shutdown()


@pytest.mark.asyncio
async def test_sync_catalog_worker_retains_original_ambient_admission():
    from kestrel_sovereign.execution_custody import (
        bind_execution_custody, current_execution_custody,
    )
    service = LLMService.__new__(LLMService)
    service._route_catalogs = None
    calls = []

    class Adapter:
        async def list_models(self):
            calls.extend(current_execution_custody())
            return []

    service._route_specific_catalog_adapters = lambda: iter([("synthetic", Adapter())])
    with bind_execution_custody(Authority()) as original:
        service._ensure_route_catalogs_sync()
    assert calls == [original], "worker dropped the original caller admission"
    assert service._route_catalogs == {"synthetic": []}


@pytest.mark.asyncio
async def test_fanout_success_rechecks_before_parent_publication(monkeypatch):
    from kestrel_sovereign import execution_custody as custody
    scope = ExecutionCustody(Authority())
    owner = SimpleNamespace(_execution_custody=scope)
    gather = asyncio.gather

    async def revoke_after_join(*tasks):
        result = await gather(*tasks)
        scope.revoke("lost before publication")
        return result

    async def complete():
        return "not publishable"

    monkeypatch.setattr(custody.asyncio if hasattr(custody, "asyncio") else asyncio, "gather", revoke_after_join)
    with pytest.raises(ExecutionAuthorityError, match="lost before publication"):
        await custody.await_execution_work_group(owner, [complete])
