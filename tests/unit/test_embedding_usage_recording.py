"""Embedding calls record usage through the shared modality recorder (#3426).

Every dispatched embedding call, successful or not, reaches the same sinks as
chat: ``model_usage``, the ``llm_calls`` row, Prometheus and the metering
callback. Records carry ``modality="embedding"``, the tokens and cost the route
reports, and no content: neither the embedded text nor a vector.
"""

from __future__ import annotations

import ast
import asyncio
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types import CreateEmbeddingResponse

import kestrel_sovereign
from kestrel_sovereign.llm.adapter import ReportedUsage
from kestrel_sovereign.llm.embedding_service import ProviderEmbeddingService
from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter
from tests.utils.process_local_llm_service import process_local_service

SECRET = "the patient said something private"
ROUTE = "openai:api"
MODEL = "text-embedding-3-small"


class UsageReportingAdapter:
    """An embedding adapter that reports provider usage, like the OpenAI one."""

    def __init__(
        self,
        *,
        vector: Optional[List[float]] = None,
        batch: Optional[List[Optional[List[float]]]] = None,
        error: Optional[BaseException] = None,
        delay: float = 0.0,
        tokens: Optional[int] = 7,
        cost: Optional[float] = 0.0003,
    ):
        self.vector = [0.1, 0.2, 0.3] if vector is None else vector
        self.batch = batch
        self.error = error
        self.delay = delay
        self.tokens = tokens
        self.cost = cost
        self.started = asyncio.Event()
        self.sinks: List[Any] = []
        self.texts: List[Any] = []
        self.during_call = None

    async def _respond(self, payload: Any, usage_sink: Optional[ReportedUsage]) -> None:
        self.texts.append(payload)
        self.sinks.append(usage_sink)
        self.started.set()
        if self.during_call is not None:
            self.during_call()
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if usage_sink is not None:
            usage_sink.add(input_tokens=self.tokens, cost=self.cost)

    async def aembed(self, client, text, *, model=None, dimensions=None,
                     usage_sink: Optional[ReportedUsage] = None, **kwargs):
        await self._respond(text, usage_sink)
        return self.vector

    async def aembed_batch(self, client, texts, *, model=None, dimensions=None,
                           usage_sink: Optional[ReportedUsage] = None, **kwargs):
        await self._respond(texts, usage_sink)
        if self.batch is not None:
            return self.batch
        return [list(self.vector) for _ in texts]


class SinklessAdapter:
    """A third-party adapter written against the bare SDK embed contract."""

    def __init__(self):
        self.kwargs: List[Dict[str, Any]] = []

    async def aembed(self, client, text, *, model=None, **kwargs):
        self.kwargs.append(kwargs)
        return [0.5, 0.5]

    async def aembed_batch(self, client, texts, *, model=None, **kwargs):
        self.kwargs.append(kwargs)
        return [[0.5, 0.5] for _ in texts]


def _provider(adapter: Any, *, name: str = ROUTE, model: str = MODEL) -> Dict[str, Any]:
    vendor, _, route = name.partition(":")
    return {
        "name": name,
        "vendor": vendor,
        "route": route,
        "adapter": adapter,
        "client": object(),
        "model": "auto",
        "is_local": False,
        "is_cloud": True,
        "capabilities": {
            "supports_embeddings": True,
            "embedding_model": model,
            "embedding_dim": 3,
        },
    }


def _llm_service(*providers: Dict[str, Any]):
    service = process_local_service(list(providers))
    service._observability_store = MagicMock()
    service._observability_store.log_llm_call = AsyncMock()
    service._owner_agent_did = "did:example:agent"
    service._track_model_usage = AsyncMock()
    return service


def _embedding_service(adapter: Any, **provider_kwargs: Any):
    llm = _llm_service(_provider(adapter, **provider_kwargs))
    embed = llm.get_embedding_service()
    assert isinstance(embed, ProviderEmbeddingService)
    return llm, embed


def _logged(service) -> List[Dict[str, Any]]:
    return [call.kwargs for call in service._observability_store.log_llm_call.await_args_list]


def _bill_as(service, *, session_id: str = "s") -> None:
    service._resolve_invocation_context = lambda *a, **k: LLMInvocationContext(
        session_id=session_id, companion_id="comp", user_id="user"
    )


# ---------------------------------------------------------------------------
# Wiring: the services LLMService builds carry its recorder
# ---------------------------------------------------------------------------


def test_every_llm_service_embedding_builder_attaches_the_recorder() -> None:
    adapter = UsageReportingAdapter()
    llm = _llm_service(_provider(adapter))
    assert llm.get_embedding_service()._recorder is llm
    assert llm.get_embedding_service_for_route(ROUTE)._recorder is llm
    assert llm._new_embedding_service(_provider(adapter))._recorder is llm


def test_production_code_builds_embedding_services_only_through_the_factory() -> None:
    """A service built anywhere else would dispatch calls nothing records."""
    package = Path(kestrel_sovereign.__file__).parent
    constructions = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {
            child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "ProviderEmbeddingService":
                continue
            owner: ast.AST = node
            while owner in parents and not isinstance(
                owner, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                owner = parents[owner]
            constructions.append(
                (path.relative_to(package).as_posix(), getattr(owner, "name", "<module>"))
            )
    assert constructions == [("llm/service.py", "_new_embedding_service")]


# ---------------------------------------------------------------------------
# What a record carries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_embedding_records_usage_without_content() -> None:
    adapter = UsageReportingAdapter(tokens=7, cost=0.0003)
    llm, embed = _embedding_service(adapter)

    vector = await embed.aembed(SECRET)

    assert vector == [0.1, 0.2, 0.3]
    [row] = _logged(llm)
    assert row["provider"] == ROUTE and row["model"] == MODEL and row["success"] is True
    assert row["metadata"]["modality"] == "embedding"
    assert row["metadata"]["operation"] == "document"
    assert row["metadata"]["input_count"] == 1
    assert row["metadata"]["usage_available"] is True
    assert row["metadata"]["provider_reported_cost_usd"] == pytest.approx(0.0003)
    assert row["input_tokens"] == 7 and row["output_tokens"] is None
    assert row["user_prompt"] is None and row["response"] is None
    assert row["system_prompt"] is None and row["error_message"] is None
    assert SECRET not in repr(row) and "0.1" not in repr(row["metadata"])
    llm._track_model_usage.assert_awaited_once_with(MODEL, ROUTE, tokens=7)


@pytest.mark.asyncio
async def test_query_and_batch_operations_are_recorded() -> None:
    adapter = UsageReportingAdapter(tokens=3, cost=None)
    llm, embed = _embedding_service(adapter)

    await embed.aembed_query("what did I say?")
    await embed.aembed_batch(["a", "b", "c"])

    query, batch = _logged(llm)
    assert query["metadata"]["operation"] == "query"
    assert query["metadata"]["input_count"] == 1
    assert "provider_reported_cost_usd" not in query["metadata"]
    assert batch["metadata"]["operation"] == "batch"
    assert batch["metadata"]["input_count"] == 3
    assert batch["success"] is True and batch["input_tokens"] == 3


@pytest.mark.asyncio
async def test_an_adapter_that_reports_nothing_records_unknown_usage() -> None:
    adapter = SinklessAdapter()
    llm, embed = _embedding_service(adapter)

    assert await embed.aembed("text") == [0.5, 0.5]

    # The sink is never forwarded to an adapter that did not name it: it could
    # pass the unknown keyword on into its own HTTP request.
    assert adapter.kwargs == [{"dimensions": 3}]
    [row] = _logged(llm)
    assert row["success"] is True and row["input_tokens"] is None
    assert row["metadata"]["usage_available"] is False
    llm._track_model_usage.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_embedding_is_recorded_and_reraised_without_its_message() -> None:
    adapter = UsageReportingAdapter(error=RuntimeError(f"HTTP 400: input was {SECRET}"))
    llm, embed = _embedding_service(adapter)

    with pytest.raises(RuntimeError):
        await embed.aembed(SECRET)

    [row] = _logged(llm)
    assert row["success"] is False and row["error_message"] == "RuntimeError"
    assert row["metadata"]["missing_vectors"] == 1
    assert row["metadata"]["usage_available"] is False
    assert SECRET not in repr(row)
    llm._track_model_usage.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_adapter_that_swallows_its_failure_is_recorded_as_failed() -> None:
    adapter = UsageReportingAdapter()
    adapter.vector = None  # what the Ollama adapter returns after a logged failure
    llm, embed = _embedding_service(adapter)

    assert await embed.aembed("text") is None

    [row] = _logged(llm)
    assert row["success"] is False and row["error_message"] is None
    assert row["metadata"]["missing_vectors"] == 1


@pytest.mark.asyncio
async def test_a_partial_batch_is_recorded_as_failed_with_its_usage() -> None:
    adapter = UsageReportingAdapter(batch=[[0.1], None, [0.3]], tokens=9)
    llm, embed = _embedding_service(adapter)

    assert await embed.aembed_batch(["a", "b", "c"]) == [[0.1], None, [0.3]]

    [row] = _logged(llm)
    assert row["success"] is False and row["metadata"]["missing_vectors"] == 1
    # The provider still reported (and billed) the tokens it consumed.
    llm._track_model_usage.assert_awaited_once_with(MODEL, ROUTE, tokens=9)


@pytest.mark.asyncio
async def test_array_vectors_do_not_break_the_record() -> None:
    np = pytest.importorskip("numpy")
    adapter = UsageReportingAdapter(batch=[np.array([0.1, 0.2]), np.array([0.3, 0.4])])
    llm, embed = _embedding_service(adapter)

    result = await embed.aembed_batch(["a", "b"])

    assert len(result) == 2
    [row] = _logged(llm)
    assert row["success"] is True and "missing_vectors" not in row["metadata"]


@pytest.mark.asyncio
async def test_an_empty_batch_dispatches_and_records_nothing() -> None:
    adapter = UsageReportingAdapter()
    llm, embed = _embedding_service(adapter)

    assert await embed.aembed_batch([]) == []
    assert _logged(llm) == []
    assert adapter.sinks == [None]


@pytest.mark.asyncio
async def test_cancellation_propagates_and_the_record_survives() -> None:
    adapter = UsageReportingAdapter(delay=30)
    llm, embed = _embedding_service(adapter)

    task = asyncio.create_task(embed.aembed("text"))
    await asyncio.wait_for(adapter.started.wait(), 30)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await llm.drain_modality_records()

    [row] = _logged(llm)
    assert row["success"] is False and row["error_message"] == "CancelledError"


@pytest.mark.asyncio
async def test_the_invocation_context_is_frozen_before_the_call() -> None:
    adapter = UsageReportingAdapter()
    llm, embed = _embedding_service(adapter)
    llm.set_observability_context(session_id="before", companion_id="c", user_id="u")
    adapter.during_call = lambda: llm.set_observability_context(
        session_id="after", companion_id="c2", user_id="u2"
    )

    await embed.aembed("text")

    [row] = _logged(llm)
    assert row["session_id"] == "before" and row["companion_id"] == "c"


@pytest.mark.asyncio
async def test_a_service_without_a_recorder_records_nothing() -> None:
    adapter = UsageReportingAdapter()
    embed = ProviderEmbeddingService(_provider(adapter))

    assert await embed.aembed("text") == [0.1, 0.2, 0.3]
    assert adapter.sinks == [None]


# ---------------------------------------------------------------------------
# Sinks shared with chat: Prometheus and metering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embeddings_count_in_the_llm_series_not_the_decision_series() -> None:
    from kestrel_sdk import metrics

    if not metrics.PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus-client not installed")
    model = "embed-metrics-3426"
    llm, embed = _embedding_service(UsageReportingAdapter(tokens=11), model=model)

    def sample(name: str, labels: Dict[str, str]) -> float:
        return metrics.REGISTRY.get_sample_value(name, labels) or 0.0

    calls = {"provider": ROUTE, "model": model, "success": "True"}
    tokens = {"model": model, "direction": "input"}
    decisions = {"model": model}
    before = (
        sample("kestrel_llm_calls_total", calls),
        sample("kestrel_llm_tokens_total", tokens),
        sample("kestrel_llm_decision_tokens_total", decisions),
    )
    await embed.aembed("text")
    assert sample("kestrel_llm_calls_total", calls) == before[0] + 1
    assert sample("kestrel_llm_tokens_total", tokens) == before[1] + 11
    assert sample("kestrel_llm_decision_tokens_total", decisions) == before[2]


@pytest.mark.asyncio
async def test_original_signature_metering_callback_bills_embeddings() -> None:
    llm, embed = _embedding_service(UsageReportingAdapter(tokens=7))
    billed: List[Dict[str, Any]] = []

    async def original(*, companion_id, user_id, provider, model, prompt_tokens,
                       completion_tokens):
        billed.append(dict(provider=provider, model=model, prompt_tokens=prompt_tokens,
                           completion_tokens=completion_tokens))

    llm.set_metering_callback(original)
    _bill_as(llm)
    await embed.aembed("text")
    assert billed == [dict(provider=ROUTE, model=MODEL, prompt_tokens=7, completion_tokens=0)]


@pytest.mark.asyncio
async def test_metering_callback_that_names_modality_and_cost_receives_them() -> None:
    llm, embed = _embedding_service(UsageReportingAdapter(tokens=7, cost=0.0003))
    seen: List[Any] = []

    async def aware(*, companion_id, user_id, provider, model, prompt_tokens,
                    completion_tokens, modality, cost):
        seen.append((modality, cost))

    llm.set_metering_callback(aware)
    _bill_as(llm)
    await embed.aembed("text")
    assert seen == [("embedding", pytest.approx(0.0003))]


@pytest.mark.asyncio
async def test_a_failed_embedding_is_not_billed() -> None:
    llm, embed = _embedding_service(UsageReportingAdapter(error=RuntimeError("503")))
    meter = AsyncMock()
    llm.set_metering_callback(meter)
    _bill_as(llm)
    with pytest.raises(RuntimeError):
        await embed.aembed("text")
    meter.assert_not_awaited()


# ---------------------------------------------------------------------------
# One recording path: a usage-DB outage suppresses no other sink
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_db_failure_does_not_suppress_the_llm_calls_row() -> None:
    llm, embed = _embedding_service(UsageReportingAdapter(tokens=7))
    llm._track_model_usage = AsyncMock(side_effect=RuntimeError("usage db down"))

    assert await embed.aembed("text") == [0.1, 0.2, 0.3]
    [row] = _logged(llm)
    assert row["input_tokens"] == 7


# ---------------------------------------------------------------------------
# Adapters fill the sink from the provider response
# ---------------------------------------------------------------------------


def _openai_response(*, n: int = 1, cost: Optional[float] = None) -> CreateEmbeddingResponse:
    usage: Dict[str, Any] = {"prompt_tokens": 5, "total_tokens": 5}
    if cost is not None:
        usage["cost"] = cost
    return CreateEmbeddingResponse.model_validate({
        "object": "list",
        "model": "qwen/qwen3-embedding-0.6b",
        "data": [
            {"object": "embedding", "index": i, "embedding": [0.1, 0.2]} for i in range(n)
        ],
        "usage": usage,
    })


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["aembed", "aembed_batch"])
async def test_openrouter_reports_tokens_and_cost(method: str) -> None:
    adapter = OpenRouterAdapter(embedding_model="qwen/qwen3-embedding-0.6b", embedding_dim=2)
    client = SimpleNamespace(embeddings=SimpleNamespace(
        create=AsyncMock(return_value=_openai_response(n=2, cost=0.00004))
    ))
    sink = ReportedUsage()
    payload = "text" if method == "aembed" else ["a", "b"]

    await getattr(adapter, method)(client, payload, usage_sink=sink)

    assert (sink.input_tokens, sink.cost) == (5, pytest.approx(0.00004))
    assert sink.model == "qwen/qwen3-embedding-0.6b"
    # The sink is consumed by the adapter, never sent to the provider.
    assert "usage_sink" not in client.embeddings.create.await_args.kwargs


@pytest.mark.asyncio
async def test_openai_reports_tokens_without_a_cost() -> None:
    adapter = OpenAIAdapter()
    client = SimpleNamespace(embeddings=SimpleNamespace(
        create=AsyncMock(return_value=_openai_response())
    ))
    sink = ReportedUsage()
    await adapter.aembed(client, "text", usage_sink=sink)
    assert (sink.input_tokens, sink.cost) == (5, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["dict", "object"])
@pytest.mark.parametrize("method", ["aembed", "aembed_batch"])
async def test_ollama_reports_prompt_eval_count(shape: str, method: str) -> None:
    import ollama

    body = {"model": "nomic-embed-text", "embeddings": [[0.1, 0.2]], "prompt_eval_count": 4}
    response = body if shape == "dict" else ollama.EmbedResponse(**body)
    client = SimpleNamespace(embed=AsyncMock(return_value=response))
    sink = ReportedUsage()
    payload = "text" if method == "aembed" else ["text"]

    await getattr(OllamaAdapter(), method)(client, payload, usage_sink=sink)

    assert (sink.input_tokens, sink.cost, sink.model) == (4, None, "nomic-embed-text")
    assert "usage_sink" not in client.embed.await_args.kwargs


@pytest.mark.asyncio
async def test_real_adapters_receive_the_sink_from_the_service() -> None:
    adapter = OpenRouterAdapter(embedding_model="qwen/qwen3-embedding-0.6b", embedding_dim=2)
    client = SimpleNamespace(embeddings=SimpleNamespace(
        create=AsyncMock(return_value=_openai_response(cost=0.00004))
    ))
    provider = _provider(adapter, name="openrouter:api", model="qwen/qwen3-embedding-0.6b")
    provider["client"] = client
    llm = _llm_service(provider)

    await llm.get_embedding_service().aembed("text")

    [row] = _logged(llm)
    assert row["input_tokens"] == 5
    assert row["metadata"]["provider_reported_cost_usd"] == pytest.approx(0.00004)


def test_reported_usage_accumulates_and_ignores_implausible_values() -> None:
    usage = ReportedUsage()
    usage.add(input_tokens=True, cost=True, model="")
    usage.add(input_tokens=-1, cost=math.nan)
    usage.add(input_tokens="5", cost=-0.1)
    assert (usage.input_tokens, usage.cost, usage.model) == (None, None, None)
    usage.add(input_tokens=3, cost=0.25, model="m")
    usage.add(input_tokens=4, cost=0.5)
    assert (usage.input_tokens, usage.cost, usage.model) == (7, 0.75, "m")
