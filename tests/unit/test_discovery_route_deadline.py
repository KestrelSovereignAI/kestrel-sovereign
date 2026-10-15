"""#3646: a route that never answers must not stall model discovery.

Model discovery had no deadline anywhere. A route that accepted the connection
and never answered (an Ollama daemon whose model slot another process held)
stalled ``discover_all_models`` in embedding reconciliation, and with it
``!model-list`` and every turn whose route still resolved ``auto``. Nothing
returned until the process was killed.

Each route's discovery steps (chat catalog, embedding discovery, decision
discovery) now run under ``[llm] discovery_route_deadline_seconds``. An
overrun is that vendor's discovery failure and the rest of discovery returns;
within one call the silent route is not asked again.

Most tests pin the loop clock and move it one deadline once the silent step
is parked, so a call that waited on a second step would never finish and the
hang guard would name it. The end-to-end tests drive a real agent and the real
``ollama`` client against a socket that accepts connections and never answers
(the #3269 fixture), on the wall clock with a short deadline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import httpx
import ollama
import pytest

from kestrel_sdk.llm.decisions import DecisionModelInfo
from kestrel_sovereign.agent import token_counter
from kestrel_sovereign.features.health.checks import check_model_discovery
from kestrel_sovereign.llm import model_catalog
from kestrel_sovereign.llm.decisions.config import (
    DecisionRouteConfig,
    parse_service_decision_config,
)
from kestrel_sovereign.llm.decisions.resolve import PinStatus, RouteDecisionState
from kestrel_sovereign.llm.discovery_deadline import (
    DEFAULT_DISCOVERY_ROUTE_DEADLINE_SECONDS,
    DISCOVERY_ROUTE_DEADLINE_KEY,
    DiscoveryStepTimeout,
    discovery_route_deadline_seconds,
)
from kestrel_sovereign.llm.embedding_discovery import EmbeddingModelInfo
from kestrel_sovereign.llm.google_adapter import GoogleAdapter
from kestrel_sovereign.llm.model_cache import (
    DiscoveryStatus,
    SharedModelCache,
    VendorDiscoveryOutcome,
    get_shared_model_cache,
)
from kestrel_sovereign.llm.model_catalog import ModelCatalogService
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from kestrel_sovereign.llm.provider_registry import ProviderRegistry
from kestrel_sovereign.llm.service import LLMServiceError
from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter
from tests.integration.test_live_routes import silent_route  # noqa: F401 - the #3269 fixture
from tests.shared.booted_agent import booted_agent
from tests.utils.process_local_llm_service import process_local_service
from tests.utils.virtual_clock import await_within_hang_guard, pinned_loop_clock

DEADLINE = DEFAULT_DISCOVERY_ROUTE_DEADLINE_SECONDS

#: The deadline the wall-clock tests configure: short, so a test costs a
#: fraction of a second per overrun.
SHORT_DEADLINE = 0.5

DISCOVERY_LOGGER = "kestrel_sovereign.llm.model_discovery"

CHAT, EMBEDDING, DECISION = "chat catalog", "embedding discovery", "decision discovery"

#: The pinned decision model's canary: not a discovery step, but a request
#: to the same route, which a silent route must not be sent.
CANARY = "pin canary"


class _Vendor:
    """A route's adapter: each discovery step answers at once, or never.

    A step named in ``silent`` waits until it is cancelled, as a request to a
    daemon that accepted the connection and never answers does.
    """

    def __init__(self, name: str, *, silent=(), chat_error: Exception | None = None):
        self.name = name
        self.silent = set(silent)
        self.chat_error = chat_error
        self.calls: Counter = Counter()
        self.cancelled: Counter = Counter()
        #: Set once a silent step is waiting, its deadline armed.
        self.parked = asyncio.Event()

    async def _answer(self, step: str, answer: Any) -> Any:
        self.calls[step] += 1
        if step in self.silent:
            self.parked.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled[step] += 1
                raise
        return answer

    async def list_models(self, client=None) -> List[ModelInfo]:
        if self.chat_error is not None:
            self.calls[CHAT] += 1
            raise self.chat_error
        return await self._answer(CHAT, [ModelInfo(
            id=f"{self.name}-chat", provider=self.name, display_name=f"{self.name}-chat",
            category=ModelCategory.CHAT, supports_streaming=True,
        )])

    async def list_embedding_models(self, client=None) -> List[EmbeddingModelInfo]:
        return await self._answer(EMBEDDING, [
            EmbeddingModelInfo(id=f"{self.name}-embed", provider=self.name, native_dim=768),
        ])

    async def list_decision_models(self, client=None) -> List[DecisionModelInfo]:
        return await self._answer(DECISION, [DecisionModelInfo(
            id=f"{self.name}-decide", vendor="", route="", context_limit=8192,
        )])

    async def adecide(self, client, model, request, *, timeout):
        return await self._answer(
            CANARY, {"answers": {"canary": {"type": "noul", "noul": 0.9}}}
        )


def _route(vendor: _Vendor, *, local: bool = False) -> Dict[str, Any]:
    return {
        "name": f"{vendor.name}:api",
        "vendor": vendor.name,
        "route": "api",
        "model": "auto",
        "adapter": vendor,
        "client": object(),
        "is_cloud": not local,
        "is_local": local,
        "selection_hints": [],
        "capabilities": {"supports_decisions": True},
        "decision_state": RouteDecisionState(config=DecisionRouteConfig()),
    }


def _service(*vendors: _Vendor, config: Dict[str, Any] | None = None):
    """A real ``LLMService`` over ``vendors``, with an empty embedding cache."""
    service = process_local_service([_route(vendor) for vendor in vendors])
    service.config = dict(config or {})
    # Discover, rather than read the seeded "nothing new" cache.
    service._embedding_discovery_cache = None
    return service


def _route_of(service, vendor: _Vendor) -> Dict[str, Any]:
    return next(p for p in service.providers if p["vendor"] == vendor.name)


def _pin_decision_models(service) -> None:
    """Pin every route's decision model, so discovery canaries each pin."""
    for route in service.providers:
        route["decision_state"] = RouteDecisionState(
            config=DecisionRouteConfig(pin=f"{route['vendor']}-decide")
        )
    service._decision_config = parse_service_decision_config({})
    service._observability_store = SimpleNamespace(log_llm_call=AsyncMock())
    service._track_model_usage = AsyncMock()
    service._owner_agent_did = "did:test:deadline"


@pytest.fixture(autouse=True)
def isolated_catalog(tmp_path, monkeypatch) -> ModelCatalogService:
    """Discovery writes its catalog cache into ``tmp_path``, not the package."""
    catalog = ModelCatalogService(
        config_path=tmp_path / "model_catalog.toml",
        cache_path=tmp_path / "model_discovery_cache.json",
    )
    monkeypatch.setattr(model_catalog, "_catalog_service", catalog)
    monkeypatch.setattr(token_counter, "_discovered_context_limits", {})
    return catalog


async def _with_one_deadline(call, *silent, deadline: float = DEADLINE):
    """Run ``call`` on a pinned clock, moved one ``deadline`` once every
    ``silent`` route (anything with a ``parked`` event) has parked.

    Returns the call's result and the loop seconds it took. A call that waits
    on a second silent step never finishes, and the hang guard names it. So
    does one whose silent routes do not all wait at once, as a listing that
    blocks the event loop would not.
    """
    with pinned_loop_clock() as clock:
        started = clock.time()

        async def move_when_parked() -> None:
            for route in silent:
                await route.parked.wait()
            clock.advance(deadline)

        mover = asyncio.ensure_future(move_when_parked())
        try:
            result = await await_within_hang_guard(call, step="discovery with a silent route")
        finally:
            mover.cancel()
        return result, clock.time() - started


def _failures() -> Dict[str, str]:
    return get_shared_model_cache().snapshot().discovery_failures


def _deadline_warnings(caplog) -> List[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == DISCOVERY_LOGGER
        and record.levelno == logging.WARNING
        and "recorded as that vendor's discovery failure" in record.getMessage()
    ]


# --------------------------------------------------------------------------
# Each step overruns on its own, as that vendor's failure
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_silent_chat_catalog_is_the_vendors_failure_and_not_asked_again(caplog):
    caplog.set_level(logging.WARNING, logger=DISCOVERY_LOGGER)
    silent, answering = _Vendor("silent", silent={CHAT}), _Vendor("answering")
    service = _service(silent, answering)

    models, took = await _with_one_deadline(
        service.discover_all_models(use_cache=False), silent
    )

    assert took == DEADLINE
    assert [m.id for m in models] == ["answering-chat"]
    assert _failures() == {
        "silent": f"DiscoveryStepTimeout: chat catalog on silent:api did not finish within {DEADLINE:g}s",
    }
    assert get_shared_model_cache().snapshot().outcomes["answering"].status is DiscoveryStatus.OK
    assert silent.cancelled[CHAT] == 1
    # The route did not answer this call, so its later steps were not sent.
    assert silent.calls[EMBEDDING] == silent.calls[DECISION] == 0
    assert _route_of(service, silent)["model"] == "auto"
    assert _route_of(service, answering)["model"] == "answering-chat"
    assert _deadline_warnings(caplog) == [
        f"silent: chat catalog on silent:api did not finish within {DEADLINE:g}s; "
        "recorded as that vendor's discovery failure",
    ]


@pytest.mark.asyncio
async def test_a_silent_embedding_listing_times_out_on_its_own(caplog):
    caplog.set_level(logging.WARNING, logger=DISCOVERY_LOGGER)
    silent, answering = _Vendor("silent", silent={EMBEDDING}), _Vendor("answering")
    service = _service(silent, answering)

    models, took = await _with_one_deadline(
        service.discover_all_models(use_cache=False), silent
    )

    assert took == DEADLINE
    # Its catalog answered; only the embedding step overran.
    assert sorted(m.id for m in models) == ["answering-chat", "silent-chat"]
    assert _failures() == {
        "silent": f"DiscoveryStepTimeout: embedding discovery on silent:api did not finish within {DEADLINE:g}s",
    }
    assert silent.cancelled[EMBEDDING] == 1
    assert silent.calls[DECISION] == 0
    assert _route_of(service, answering)["capabilities"]["supports_embeddings"] is True
    assert "supports_embeddings" not in _route_of(service, silent)["capabilities"]
    # Resolving the answering route reused the call's one sweep, which is not
    # cached, rather than sweeping every route again.
    assert silent.calls[EMBEDDING] == answering.calls[EMBEDDING] == 1
    assert service._embedding_discovery_cache is None
    assert len(_deadline_warnings(caplog)) == 1
    assert "embedding discovery on silent:api" in _deadline_warnings(caplog)[0]


@pytest.mark.asyncio
async def test_a_silent_decision_listing_times_out_on_its_own(caplog):
    caplog.set_level(logging.WARNING, logger=DISCOVERY_LOGGER)
    silent, answering = _Vendor("silent", silent={DECISION}), _Vendor("answering")
    service = _service(silent, answering)

    models, took = await _with_one_deadline(
        service.discover_all_models(use_cache=False), silent
    )

    assert took == DEADLINE
    assert sorted(m.id for m in models) == ["answering-chat", "silent-chat"]
    assert _failures() == {
        "silent": f"DiscoveryStepTimeout: decision discovery on silent:api did not finish within {DEADLINE:g}s",
    }
    assert silent.cancelled[DECISION] == 1
    silent_state = _route_of(service, silent)["decision_state"]
    assert silent_state.models is None
    assert silent_state.discovery_stale_since is not None
    assert [m.id for m in _route_of(service, answering)["decision_state"].models] == [
        "answering-decide"
    ]
    # Both other steps of the silent route answered.
    assert _route_of(service, silent)["capabilities"]["supports_embeddings"] is True
    assert len(_deadline_warnings(caplog)) == 1
    assert "decision discovery on silent:api" in _deadline_warnings(caplog)[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("silent_step", [CHAT, EMBEDDING, DECISION])
async def test_a_pinned_route_that_did_not_answer_gets_no_canary(silent_step):
    """A pinned decision model is checked with one canary decision after the
    listing. A route that did not answer this call, at its decision listing or
    at an earlier step that skipped it, is not sent one: the canary would wait
    out its own timeout on top of the deadline. Its pin keeps its last
    verification, marked stale. The answering route's pin is still checked.

    The silent route never answers its canary either, so sending one would
    park the call past the single deadline the clock is moved by.
    """
    silent = _Vendor("silent", silent={silent_step, DECISION, CANARY})
    answering = _Vendor("answering")
    service = _service(silent, answering)
    _pin_decision_models(service)

    _, took = await _with_one_deadline(service.discover_all_models(use_cache=False), silent)

    assert took == DEADLINE
    assert silent.calls[DECISION] == (1 if silent_step == DECISION else 0)
    assert silent.calls[CANARY] == 0
    silent_state = _route_of(service, silent)["decision_state"]
    assert silent_state.pin_status is PinStatus.UNCHECKED
    assert silent_state.discovery_stale_since is not None
    assert silent_state.canary_stale_since is not None
    assert _failures() == {
        "silent": f"DiscoveryStepTimeout: {silent_step} on silent:api did not finish within {DEADLINE:g}s",
    }

    assert answering.calls[CANARY] == 1
    answering_state = _route_of(service, answering)["decision_state"]
    assert answering_state.pin_status is PinStatus.VERIFIED
    assert answering_state.canary_stale_since is None


@pytest.mark.asyncio
async def test_a_pinned_route_that_answers_is_canaried_on_every_refresh():
    answering = _Vendor("answering")
    service = _service(answering)
    _pin_decision_models(service)

    for refresh in (1, 2):
        await await_within_hang_guard(
            service.discover_all_models(use_cache=False), step="discovery"
        )
        assert answering.calls[CANARY] == refresh
    assert _route_of(service, answering)["decision_state"].pin_status is PinStatus.VERIFIED


@pytest.mark.asyncio
async def test_a_task_created_during_a_call_does_not_inherit_its_silence():
    """A task created during a discovery call copies its context, and can
    outlive it. Its own later discovery asks every route again, rather than
    skipping one the ended call found silent."""
    silent, answering = _Vendor("silent", silent={CHAT}), _Vendor("answering")
    service = _service(silent, answering)
    later = asyncio.Event()
    spawned: List[asyncio.Future] = []
    list_models = answering.list_models

    async def list_and_spawn(client=None):
        if not spawned:
            async def discover_later():
                await later.wait()
                return await service.discover_all_models(use_cache=False)

            spawned.append(asyncio.ensure_future(discover_later()))
        return await list_models(client)

    answering.list_models = list_and_spawn

    _, took = await _with_one_deadline(service.discover_all_models(use_cache=False), silent)
    assert took == DEADLINE
    assert silent.calls[CHAT] == 1

    silent.silent.clear()
    later.set()
    models = await await_within_hang_guard(spawned[0], step="a later discovery")

    assert silent.calls[CHAT] == 2
    assert sorted(m.id for m in models) == ["answering-chat", "silent-chat"]


@pytest.mark.asyncio
async def test_a_task_created_during_a_pass_records_its_later_overrun(caplog):
    """A task created during a refresh holds the refresh's pass in its
    context. Once that pass has published, the task's later discovery is
    outside any pass: an overrun there amends the current catalog, instead of
    going into the ended pass's map, which nothing will publish again."""
    caplog.set_level(logging.WARNING, logger=DISCOVERY_LOGGER)
    late = _Vendor("late")
    service = _service(late)
    later = asyncio.Event()
    spawned: List[asyncio.Future] = []
    list_models = late.list_models

    async def list_and_spawn(client=None):
        if not spawned:
            async def reconcile_later():
                await later.wait()
                await service.reconcile_decision_capabilities(use_cache=False)

            spawned.append(asyncio.ensure_future(reconcile_later()))
        return await list_models(client)

    late.list_models = list_and_spawn
    await await_within_hang_guard(service.discover_all_models(use_cache=False), step="refresh")
    assert _failures() == {}

    late.silent.add(DECISION)
    later.set()
    _, took = await _with_one_deadline(spawned[0], late)

    assert took == DEADLINE
    assert _failures() == {
        "late": f"DiscoveryStepTimeout: decision discovery on late:api did not finish within {DEADLINE:g}s",
    }
    assert _deadline_warnings(caplog) == [
        f"late: decision discovery on late:api did not finish within {DEADLINE:g}s; "
        "recorded as that vendor's discovery failure",
    ]


@pytest.mark.asyncio
async def test_a_steps_own_timeout_counts_as_its_overrun(caplog):
    """The Ollama adapter bounds its listings by the same value; whichever
    bound fires first, the route's catalog is recorded as having timed out
    and the route is not asked again."""
    caplog.set_level(logging.WARNING, logger=DISCOVERY_LOGGER)
    own_bound = _Vendor("own", chat_error=TimeoutError("Ollama model listing did not answer"))
    service = _service(own_bound)

    await await_within_hang_guard(
        service.discover_all_models(use_cache=False), step="discovery"
    )

    assert _failures() == {
        "own": "DiscoveryStepTimeout: chat catalog on own:api timed out: "
        "Ollama model listing did not answer",
    }
    assert own_bound.calls[EMBEDDING] == own_bound.calls[DECISION] == 0
    assert len(_deadline_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_the_configured_deadline_is_the_one_applied():
    silent = _Vendor("silent", silent={CHAT})
    service = _service(silent, config={DISCOVERY_ROUTE_DEADLINE_KEY: 2.5})

    _, took = await _with_one_deadline(
        service.discover_all_models(use_cache=False), silent, deadline=2.5
    )

    assert took == 2.5
    assert _failures()["silent"].endswith("did not finish within 2.5s")


@pytest.mark.asyncio
async def test_a_later_overrun_does_not_replace_the_vendors_first_failure():
    silent = _Vendor("silent", silent={EMBEDDING}, chat_error=RuntimeError("401 Unauthorized"))
    service = _service(silent)

    _, took = await _with_one_deadline(service.discover_all_models(use_cache=False), silent)

    assert took == DEADLINE
    assert _failures() == {"silent": "RuntimeError: 401 Unauthorized"}


def _chat_model(model_id: str, vendor: str) -> ModelInfo:
    return ModelInfo(
        id=model_id, provider=vendor, display_name=model_id,
        category=ModelCategory.CHAT, supports_streaming=True,
    )


@pytest.mark.asyncio
async def test_a_cache_hits_overrun_reaches_the_health_check_with_the_catalog_it_served(caplog):
    """A turn's warm-up mostly hits the cache, then reconciles every route.

    A fresh service reading the process-wide catalog still contacts its
    routes' embedding listings. An overrun there is the vendor's failure: the
    snapshot the hit served gains it, with its catalog and age unchanged, and
    the ``model_discovery`` health check reports it. Before this, the warning
    claimed the failure was recorded while the snapshot kept reporting none.
    """
    caplog.set_level(logging.WARNING, logger=DISCOVERY_LOGGER)
    silent = _Vendor("silent", silent={EMBEDDING})
    answering = _Vendor("answering")
    service = _service(silent, answering)
    cache = get_shared_model_cache()
    cache.set(
        [_chat_model("silent-chat", "silent"), _chat_model("answering-chat", "answering")],
        outcomes={"silent": VendorDiscoveryOutcome.ok(), "answering": VendorDiscoveryOutcome.ok()},
    )
    served = cache.snapshot()

    models, took = await _with_one_deadline(
        service.discover_all_models(use_cache=True), silent
    )

    assert took == DEADLINE
    assert sorted(m.id for m in models) == ["answering-chat", "silent-chat"]
    assert silent.cancelled[EMBEDDING] == 1
    assert silent.calls[DECISION] == 0
    assert _route_of(service, answering)["capabilities"]["supports_embeddings"] is True

    failure = f"DiscoveryStepTimeout: embedding discovery on silent:api did not finish within {DEADLINE:g}s"
    amended = cache.snapshot()
    assert amended.version == served.version + 1
    assert amended.catalog_version == served.catalog_version
    assert (amended.models, amended.timestamp) == (served.models, served.timestamp)
    assert amended.discovery_failures == {"silent": failure}
    assert amended.outcomes["silent"].observed_version == amended.version
    assert amended.outcomes["answering"] == served.outcomes["answering"]
    assert len(_deadline_warnings(caplog)) == 1

    health = await check_model_discovery(SimpleNamespace(llm_service=service))
    assert health["status"] == "warn"
    assert health["details"]["failed_vendors"] == {"silent": failure}
    assert health["details"]["catalog_version"] == served.catalog_version

    # Another service's hit finds the failure already reported: nothing new.
    silent.parked.clear()
    await _with_one_deadline(_service(silent, answering).discover_all_models(), silent)
    assert cache.snapshot() is amended


@pytest.mark.asyncio
async def test_a_cache_hits_overrun_is_not_pinned_on_a_catalog_published_since():
    """The failure belongs to the catalog the hit served. A refresh that
    published while the route was silent observed its vendors itself."""
    silent = _Vendor("silent", silent={EMBEDDING})
    service = _service(silent)
    cache = get_shared_model_cache()
    cache.set([_chat_model("silent-chat", "silent")], outcomes={"silent": VendorDiscoveryOutcome.ok()})

    async def refresh_while_parked() -> None:
        await silent.parked.wait()
        cache.set([_chat_model("silent-chat", "silent")], outcomes={"silent": VendorDiscoveryOutcome.ok()})

    refresher = asyncio.ensure_future(refresh_while_parked())
    try:
        await _with_one_deadline(service.discover_all_models(use_cache=True), silent)
    finally:
        refresher.cancel()

    assert cache.snapshot().discovery_failures == {}
    assert cache.snapshot().outcomes["silent"].status is DiscoveryStatus.OK


def _timeout(vendor: str, note: str = "") -> VendorDiscoveryOutcome:
    return VendorDiscoveryOutcome.failed(f"DiscoveryStepTimeout: {vendor}{note}")


def test_amendments_against_one_catalog_merge_and_keep_each_vendors_latest_failure():
    """Services reconciling against one catalog each add what they saw.

    Each amendment republishes the snapshot, so a version check would reject
    every amendment after the first made against the same catalog, and a real
    timeout would vanish from the health check. Only the catalog matters.
    """
    cache = SharedModelCache()
    cache.set([_chat_model("a-chat", "a")], outcomes={
        "a": VendorDiscoveryOutcome.ok(),
        "b": VendorDiscoveryOutcome.failed("RuntimeError: 401"),
    })
    served = cache.snapshot()

    assert cache.record_failures(served.catalog_version, {"a": _timeout("a")})
    assert cache.record_failures(served.catalog_version, {"c": _timeout("c")})
    merged = cache.snapshot()
    assert merged.version == served.version + 2
    assert merged.catalog_version == served.catalog_version
    assert (merged.models, merged.timestamp) == (served.models, served.timestamp)
    assert merged.discovery_failures == {
        "a": "DiscoveryStepTimeout: a",
        "b": "RuntimeError: 401",
        "c": "DiscoveryStepTimeout: c",
    }

    # A vendor reported failing again reports the later failure.
    assert cache.record_failures(served.catalog_version, {"b": _timeout("b")})
    assert cache.snapshot().discovery_failures["b"] == "DiscoveryStepTimeout: b"
    assert cache.snapshot().outcomes["b"].observed_version == cache.snapshot().version

    # The same failure again adds nothing and does not republish.
    latest = cache.snapshot()
    assert cache.record_failures(served.catalog_version, {"a": _timeout("a")})
    assert cache.snapshot() is latest

    with pytest.raises(ValueError, match="only a failed outcome"):
        cache.record_failures(served.catalog_version, {"c": VendorDiscoveryOutcome.ok()})


@pytest.mark.parametrize("replace", ["refresh", "restore", "invalidate"])
def test_an_amendment_against_a_replaced_catalog_is_rejected(replace):
    cache = SharedModelCache()
    cache.set([_chat_model("a-chat", "a")], outcomes={"a": VendorDiscoveryOutcome.ok()})
    served = cache.snapshot()
    assert cache.record_failures(served.catalog_version, {"a": _timeout("a")})

    if replace == "refresh":
        cache.set([_chat_model("a-chat", "a")], outcomes={"a": VendorDiscoveryOutcome.ok()})
    elif replace == "restore":
        cache.clear()
        cache.set_stale([_chat_model("a-chat", "a")])
    else:
        cache.clear()
    replaced = cache.snapshot()
    assert replaced.catalog_version > served.catalog_version

    assert not cache.record_failures(served.catalog_version, {"b": _timeout("b")})
    assert cache.snapshot() is replaced


@pytest.mark.asyncio
async def test_two_services_failures_against_one_catalog_both_reach_the_health_check():
    """Two agents' warm-ups hit the same catalog, each with its own silent route.

    Both read the catalog before either amends it; the second amendment used
    to be rejected because the first had republished the snapshot, and its
    vendor's timeout never reached the health check.
    """
    first_silent = _Vendor("first", silent={EMBEDDING})
    second_silent = _Vendor("second", silent={EMBEDDING})
    first, second = _service(first_silent), _service(second_silent)
    cache = get_shared_model_cache()
    cache.set(
        [_chat_model("first-chat", "first"), _chat_model("second-chat", "second")],
        outcomes={"first": VendorDiscoveryOutcome.ok(), "second": VendorDiscoveryOutcome.ok()},
    )
    served = cache.snapshot()

    async def both_hit_the_cache():
        return await asyncio.gather(first.discover_all_models(), second.discover_all_models())

    _, took = await _with_one_deadline(both_hit_the_cache(), first_silent, second_silent)

    assert took == DEADLINE
    failures = {
        name: f"DiscoveryStepTimeout: embedding discovery on {name}:api did not finish within {DEADLINE:g}s"
        for name in ("first", "second")
    }
    amended = cache.snapshot()
    assert amended.catalog_version == served.catalog_version
    assert amended.models == served.models
    assert amended.discovery_failures == failures
    for service in (first, second):
        health = await check_model_discovery(SimpleNamespace(llm_service=service))
        assert health["details"]["failed_vendors"] == failures
        assert health["details"]["catalog_version"] == served.catalog_version


@pytest.mark.asyncio
async def test_local_only_auto_resolution_is_bounded_and_amends_the_current_catalog(caplog):
    """A privacy-gated turn resolves its local routes without the full pass.

    It publishes no catalog, so the overrun amends the one that is current,
    as a cache hit's would.
    """
    caplog.set_level(logging.WARNING, logger=DISCOVERY_LOGGER)
    silent = _Vendor("silent", silent={CHAT})
    service = process_local_service([_route(silent, local=True)])
    cache = get_shared_model_cache()
    cache.set([_chat_model("other-chat", "other")], outcomes={"other": VendorDiscoveryOutcome.ok()})
    served = cache.snapshot()

    _, took = await _with_one_deadline(service._resolve_local_auto_routes(), silent)

    assert took == DEADLINE
    assert service.providers[0]["model"] == "auto"
    assert cache.snapshot().catalog_version == served.catalog_version
    assert _failures() == {
        "silent": f"DiscoveryStepTimeout: chat catalog on silent:api did not finish within {DEADLINE:g}s",
    }
    assert _deadline_warnings(caplog) == [
        f"silent: chat catalog on silent:api did not finish within {DEADLINE:g}s; "
        "recorded as that vendor's discovery failure",
    ]


@pytest.mark.asyncio
async def test_an_overrun_with_no_catalog_published_says_it_was_not_recorded(caplog):
    caplog.set_level(logging.WARNING, logger=DISCOVERY_LOGGER)
    silent = _Vendor("silent", silent={CHAT})
    service = process_local_service([_route(silent, local=True)])

    _, took = await _with_one_deadline(service._resolve_local_auto_routes(), silent)

    assert took == DEADLINE
    assert get_shared_model_cache().snapshot() is None
    assert [r.getMessage() for r in caplog.records if r.name == DISCOVERY_LOGGER] == [
        f"silent: chat catalog on silent:api did not finish within {DEADLINE:g}s; "
        "no model catalog has been published, so not recorded",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("step, reconcile", [
    (EMBEDDING, lambda service: service.reconcile_embedding_capabilities(use_cache=True)),
    (DECISION, lambda service: service.reconcile_decision_capabilities(use_cache=False)),
])
async def test_a_reconcile_called_on_its_own_records_its_overrun(step, reconcile, caplog):
    """The embedding settings endpoints, boot and ``kestrel decisions``
    reconcile outside any discovery pass. A route that overruns there is
    still that vendor's failure, against the catalog current when it began."""
    caplog.set_level(logging.WARNING, logger=DISCOVERY_LOGGER)
    silent, answering = _Vendor("silent", silent={step}), _Vendor("answering")
    service = _service(silent, answering)
    cache = get_shared_model_cache()
    cache.set(
        [_chat_model("silent-chat", "silent"), _chat_model("answering-chat", "answering")],
        outcomes={"silent": VendorDiscoveryOutcome.ok(), "answering": VendorDiscoveryOutcome.ok()},
    )
    served = cache.snapshot()

    _, took = await _with_one_deadline(reconcile(service), silent)

    assert took == DEADLINE
    assert silent.cancelled[step] == 1
    # Resolving the answering route reused the sweep: nothing was asked twice.
    assert silent.calls[step] == 1
    assert answering.calls[step] == 1
    assert cache.snapshot().catalog_version == served.catalog_version
    assert _failures() == {
        "silent": f"DiscoveryStepTimeout: {step} on silent:api did not finish within {DEADLINE:g}s",
    }
    assert len(_deadline_warnings(caplog)) == 1


# --------------------------------------------------------------------------
# Google and Vertex listings await, so the deadline can cut them off
# --------------------------------------------------------------------------


class _BlockingSyncTransport(httpx.BaseTransport):
    """The synchronous side of a silent endpoint: it holds the calling thread.

    Bounded, so a regression to a synchronous listing blocks the event loop
    for a moment and then fails, instead of reaching the network or hanging
    the run.
    """

    def __init__(self) -> None:
        self.paths: List[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        time.sleep(0.2)
        raise httpx.ReadTimeout("a silent endpoint never answers", request=request)


class _SilentTransport(httpx.AsyncBaseTransport):
    """An HTTP transport that takes every request and never answers.

    ``parked`` is set once a request is waiting, like ``_Vendor.parked``.
    ``sync`` serves the client's synchronous requests.
    """

    def __init__(self) -> None:
        self.parked = asyncio.Event()
        self.paths: List[str] = []
        self.sync = _BlockingSyncTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        self.parked.set()
        await asyncio.Event().wait()
        raise AssertionError("a silent endpoint never answers")


def _genai_client(transport: _SilentTransport, **kwargs):
    """The real google-genai client, all its HTTP going to ``transport``."""
    genai = pytest.importorskip("google.genai")
    genai_types = pytest.importorskip("google.genai.types")
    if not {"httpx_client", "httpx_async_client"} <= set(genai_types.HttpOptions.model_fields):
        pytest.skip("installed google-genai cannot take httpx clients")
    return genai.Client(
        http_options=genai_types.HttpOptions(
            httpx_client=httpx.Client(transport=transport.sync),
            httpx_async_client=httpx.AsyncClient(transport=transport),
        ),
        **kwargs,
    )


def _gemini_route(vendor: str, adapter, client) -> Dict[str, Any]:
    return {
        "name": f"{vendor}:api", "vendor": vendor, "route": "api", "model": "auto",
        "adapter": adapter, "client": client, "is_cloud": True, "is_local": False,
        "selection_hints": [], "capabilities": {},
    }


def _silent_google_route(transport: _SilentTransport) -> Dict[str, Any]:
    """``google:api``: the real adapter on the client the registry gives it."""
    return _gemini_route("google", GoogleAdapter(), _genai_client(transport, api_key="test-key"))


def _silent_vertex_route(transport: _SilentTransport) -> Dict[str, Any]:
    """``vertex_ai:api``: the real adapter on its own Vertex-mode client."""
    from google.oauth2.credentials import Credentials

    adapter = VertexAIAdapter(project_id="deadline-test", location="us-central1")
    adapter._client = _genai_client(
        transport, vertexai=True, project="deadline-test", location="us-central1",
        credentials=Credentials(token="test-token"),
    )
    return _gemini_route("vertex_ai", adapter, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor, silent_route_for", [
    ("google", _silent_google_route),
    ("vertex_ai", _silent_vertex_route),
])
async def test_a_silent_gemini_listing_is_cut_off_without_blocking_other_routes(
    vendor, silent_route_for, monkeypatch,
):
    """The Gemini listings were synchronous SDK calls, which no deadline can
    cancel: a silent endpoint blocked the event loop, so every other route's
    discovery and deadline waited on it. They now await on the route's client.

    Both silent routes must be waiting at once before the clock moves, which a
    listing that blocks the loop never allows.
    """
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT_ID"):
        monkeypatch.delenv(name, raising=False)
    transport = _SilentTransport()
    other = _Vendor("other", silent={CHAT})
    answering = _Vendor("answering")
    service = _service(other, answering)
    service.providers.insert(0, silent_route_for(transport))

    models, took = await _with_one_deadline(
        service.discover_all_models(use_cache=False), transport, other
    )

    assert took == DEADLINE
    assert [m.id for m in models] == ["answering-chat"]
    assert len(transport.paths) == 1 and transport.paths[0].endswith("/models")
    assert transport.sync.paths == []
    assert _failures() == {
        vendor: f"DiscoveryStepTimeout: chat catalog on {vendor}:api did not finish within {DEADLINE:g}s",
        "other": f"DiscoveryStepTimeout: chat catalog on other:api did not finish within {DEADLINE:g}s",
    }


# --------------------------------------------------------------------------
# Coalesced embedding discovery
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coalesced_waiters_are_released_and_the_overrun_is_not_cached():
    silent, answering = _Vendor("silent", silent={EMBEDDING}), _Vendor("answering")
    service = _service(silent, answering)

    async def three_waiters():
        return await asyncio.gather(*(service.discover_embedding_models() for _ in range(3)))

    results, took = await _with_one_deadline(three_waiters(), silent)

    assert took == DEADLINE
    assert [[m.id for m in models] for models in results] == [["answering-embed"]] * 3
    # One shared sweep, not one per waiter.
    assert silent.calls[EMBEDDING] == answering.calls[EMBEDDING] == 1
    # The silent route's catalog is unknown, not empty: nothing is cached.
    assert service._embedding_discovery_cache is None

    silent.silent.clear()
    models = await service.discover_embedding_models()
    assert sorted(m.id for m in models) == ["answering-embed", "silent-embed"]
    assert silent.calls[EMBEDDING] == 2
    assert sorted(m.id for m in service._embedding_discovery_cache) == [
        "answering-embed", "silent-embed",
    ]

    # A refresh that overruns keeps the last complete catalog cached.
    silent.silent.add(EMBEDDING)
    silent.parked.clear()
    models, _ = await _with_one_deadline(
        service.discover_embedding_models(use_cache=False), silent
    )
    assert [m.id for m in models] == ["answering-embed"]
    assert sorted(m.id for m in service._embedding_discovery_cache) == [
        "answering-embed", "silent-embed",
    ]


@pytest.mark.asyncio
async def test_the_embedding_settings_read_asks_a_silent_route_once():
    """The settings read reconciles every route, then resolves the active one.
    Both are one discovery call, so the resolution reuses the reconcile's
    sweep instead of sweeping again and waiting out another deadline."""
    silent, answering = _Vendor("silent", silent={EMBEDDING}), _Vendor("answering")
    service = _service(silent, answering)
    service._embedding_route = "answering:api"

    settings, took = await _with_one_deadline(service.aget_embedding_settings(), silent)

    assert took == DEADLINE
    assert settings["resolved_route"] == "answering:api"
    assert settings["embedding_model"] == "answering-embed"
    assert silent.calls[EMBEDDING] == answering.calls[EMBEDDING] == 1


@pytest.mark.asyncio
async def test_a_pass_that_joins_another_callers_sweep_takes_on_its_overrun(caplog):
    """A refresh whose embedding reconcile joins a sweep started outside it.

    The sweep runs in its starter's context, so its overrun was recorded only
    there (here, nowhere: no catalog had been published). The refresh took the
    sweep's result and used to publish the silent vendor healthy, then send
    the silent route its decision listing.
    """
    caplog.set_level(logging.WARNING, logger=DISCOVERY_LOGGER)
    silent = _Vendor("silent", silent={EMBEDDING, DECISION})
    answering = _Vendor("answering")
    service = _service(silent, answering)

    joined = asyncio.Event()
    discover_embedding_models = service.discover_embedding_models

    async def refresh_joins_the_sweep():
        starter = asyncio.ensure_future(discover_embedding_models(use_cache=False))
        await silent.parked.wait()

        async def joining(**kwargs):
            joined.set()
            return await discover_embedding_models(**kwargs)

        service.discover_embedding_models = joining
        refresh = await service.discover_all_models(use_cache=False)
        return refresh, await starter

    with pinned_loop_clock() as clock:
        started = clock.time()

        async def move_once_joined():
            await joined.wait()
            clock.advance(DEADLINE)

        mover = asyncio.ensure_future(move_once_joined())
        try:
            (refreshed, swept) = await await_within_hang_guard(
                refresh_joins_the_sweep(), step="a refresh joining a silent sweep"
            )
        finally:
            mover.cancel()
        took = clock.time() - started

    assert took == DEADLINE
    assert [m.id for m in swept] == ["answering-embed"]
    assert sorted(m.id for m in refreshed) == ["answering-chat", "silent-chat"]
    # One sweep, reused by the refresh; it did not ask the silent route
    # anything after it.
    assert silent.calls[EMBEDDING] == answering.calls[EMBEDDING] == 1
    assert silent.calls[DECISION] == 0
    assert _failures() == {
        "silent": f"DiscoveryStepTimeout: embedding discovery on silent:api did not finish within {DEADLINE:g}s",
    }
    assert [
        r.getMessage() for r in caplog.records if r.name == DISCOVERY_LOGGER
    ] == [
        f"silent: embedding discovery on silent:api did not finish within {DEADLINE:g}s; "
        "no model catalog has been published, so not recorded",
    ]


# --------------------------------------------------------------------------
# The deadline's configuration
# --------------------------------------------------------------------------


def test_the_deadline_defaults_to_single_digit_seconds():
    assert discovery_route_deadline_seconds({}) == DEFAULT_DISCOVERY_ROUTE_DEADLINE_SECONDS
    assert discovery_route_deadline_seconds(None) == DEFAULT_DISCOVERY_ROUTE_DEADLINE_SECONDS
    assert 0 < DEFAULT_DISCOVERY_ROUTE_DEADLINE_SECONDS < 10
    assert discovery_route_deadline_seconds({DISCOVERY_ROUTE_DEADLINE_KEY: 3}) == 3.0


@pytest.mark.parametrize("value", [0, -1, "5", True, float("nan"), float("inf"), None])
def test_a_malformed_deadline_is_refused(value):
    with pytest.raises(ValueError, match=DISCOVERY_ROUTE_DEADLINE_KEY):
        discovery_route_deadline_seconds({DISCOVERY_ROUTE_DEADLINE_KEY: value})


def test_llm_service_refuses_a_malformed_deadline_at_construction(monkeypatch):
    from kestrel_sovereign.llm import service as service_module

    monkeypatch.setattr(
        service_module, "load_section",
        lambda section: {DISCOVERY_ROUTE_DEADLINE_KEY: 0} if section == "llm" else {},
    )
    with pytest.raises(ValueError, match=DISCOVERY_ROUTE_DEADLINE_KEY):
        service_module.LLMService()


def test_the_ollama_route_takes_the_deadline_and_its_chat_client_no_timeout():
    info = ProviderRegistry({DISCOVERY_ROUTE_DEADLINE_KEY: 2.0})._build_route(
        "ollama", "local", {"is_cloud": False},
        {"adapter": "OllamaAdapter", "host": "http://ollama.test:11434", "model": "auto"},
    )

    assert info.adapter._discovery_timeout_seconds == 2.0
    # Generation is bounded by its own cancellable path and watchdog; the
    # discovery bound must never become a read timeout on the chat client.
    assert info.client._client.timeout == httpx.Timeout(None)


def test_the_ollama_adapter_refuses_a_malformed_discovery_timeout():
    with pytest.raises(ValueError, match=DISCOVERY_ROUTE_DEADLINE_KEY):
        OllamaAdapter(discovery_timeout_seconds=0)


def test_a_skipped_step_says_it_was_not_sent():
    skipped = DiscoveryStepTimeout(
        vendor="ollama", route="ollama:local", step=EMBEDDING, seconds=5.0, skipped=True
    )
    assert str(skipped) == (
        "embedding discovery on ollama:local skipped: the route did not answer an "
        "earlier discovery step"
    )


# --------------------------------------------------------------------------
# The Ollama adapter against a daemon that never answers
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("listing", [
    "list_models", "list_embedding_models", "list_decision_models",
])
async def test_each_ollama_listing_is_bounded_on_the_routes_own_daemon(
    listing, silent_route,  # noqa: F811 - the imported fixture
):
    adapter = OllamaAdapter(discovery_timeout_seconds=SHORT_DEADLINE)
    client = ollama.AsyncClient(host=silent_route)

    with pytest.raises(TimeoutError, match=rf"did not answer within {SHORT_DEADLINE:g}s"):
        await await_within_hang_guard(
            getattr(adapter, listing)(client), step=f"OllamaAdapter.{listing}"
        )


# --------------------------------------------------------------------------
# A responsive daemon, and a long generation
# --------------------------------------------------------------------------


class _Daemon:
    """A stand-in Ollama daemon serving ``/api/tags``, ``/api/show`` and ``/api/chat``."""

    MODELS = {
        "qwen2.5:0.5b": {"capabilities": ["completion", "tools"], "model_info": {}},
        "nomic-embed-text:latest": {"capabilities": ["embedding"], "model_info": {}},
        "nimble:latest": {
            "capabilities": ["decision"], "parameters": "num_ctx 8194", "model_info": {},
        },
    }

    def __init__(self, *, chat_pieces=(), gap: float = 0.0) -> None:
        self.chat_pieces = list(chat_pieces)
        self.gap = gap

    def client(self) -> ollama.AsyncClient:
        return ollama.AsyncClient(
            host="http://ollama.test:11434", transport=httpx.MockTransport(self._handle)
        )

    async def _chat_body(self):
        for piece in self.chat_pieces:
            await asyncio.sleep(self.gap)
            yield (json.dumps({
                "model": "qwen2.5:0.5b",
                "message": {"role": "assistant", "content": piece},
                "done": False,
            }) + "\n").encode()
        yield (json.dumps({
            "model": "qwen2.5:0.5b",
            "message": {"role": "assistant", "content": ""},
            "done": True, "done_reason": "stop",
            "prompt_eval_count": 3, "eval_count": len(self.chat_pieces),
        }) + "\n").encode()

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": name, "model": name, "size": 1} for name in self.MODELS
            ]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json=self.MODELS[json.loads(request.content)["model"]])
        if request.url.path == "/api/chat":
            return httpx.Response(
                200, headers={"content-type": "application/x-ndjson"},
                content=self._chat_body(),
            )
        return httpx.Response(404)


@pytest.mark.asyncio
async def test_a_responsive_ollama_still_discovers_every_facet():
    daemon = _Daemon()
    service = process_local_service([{
        "name": "ollama:local", "vendor": "ollama", "route": "local", "model": "auto",
        "adapter": OllamaAdapter(discovery_timeout_seconds=SHORT_DEADLINE),
        "client": daemon.client(), "is_cloud": False, "is_local": True,
        "selection_hints": [],
        "capabilities": {"supports_decisions": True, "supports_embeddings": True},
        "decision_state": RouteDecisionState(config=DecisionRouteConfig()),
    }])
    service.config = {DISCOVERY_ROUTE_DEADLINE_KEY: SHORT_DEADLINE}
    service._embedding_discovery_cache = None

    models = await await_within_hang_guard(
        service.discover_all_models(use_cache=False), step="discovery on a responsive Ollama"
    )

    route = service.providers[0]
    assert "qwen2.5:0.5b" in [m.id for m in models]
    assert route["model"] == "qwen2.5:0.5b"
    assert get_shared_model_cache().snapshot().outcomes["ollama"].status is DiscoveryStatus.OK
    assert [m.id for m in service._embedding_discovery_cache] == ["nomic-embed-text:latest"]
    assert [m.id for m in route["decision_state"].models] == ["nimble:latest"]


@pytest.mark.asyncio
async def test_a_generation_longer_than_the_discovery_timeout_is_not_cut_off():
    """Each gap between tokens outlasts the discovery timeout; the stream completes."""
    discovery_timeout = 0.1
    pieces = ["slow ", "but ", "whole"]
    daemon = _Daemon(chat_pieces=pieces, gap=3 * discovery_timeout)
    adapter = OllamaAdapter(discovery_timeout_seconds=discovery_timeout)

    async def stream() -> List[Any]:
        return [item async for item in adapter.get_streaming_response(
            daemon.client(), "qwen2.5:0.5b", [{"role": "user", "content": "hi"}],
        )]

    started = time.monotonic()
    received = await await_within_hang_guard(stream(), step="a long Ollama stream")

    assert time.monotonic() - started >= len(pieces) * 3 * discovery_timeout
    assert "".join(item for item in received if isinstance(item, str)) == "slow but whole"


# --------------------------------------------------------------------------
# End to end: a real agent and the real ollama client on a silent daemon
# --------------------------------------------------------------------------


class _AnsweringCatalog:
    """Another vendor whose catalog answers, so the listing has something to show."""

    async def list_models(self, client=None) -> List[ModelInfo]:
        return [ModelInfo(
            id="fixture-chat", provider="fixture", display_name="fixture-chat",
            category=ModelCategory.CHAT, supports_streaming=True,
        )]


def _silent_ollama_route(service, url: str) -> Dict[str, Any]:
    """``ollama:local`` built as the provider registry builds it, at ``url``."""
    info = ProviderRegistry(service.config)._build_route(
        "ollama", "local", {"is_cloud": False},
        {"adapter": "OllamaAdapter", "host": url, "model": "auto", "local": True},
    )
    [route] = service._convert_providers_format([info])
    assert route["client"]._client.base_url == httpx.URL(url)
    return route


async def _on_a_silent_ollama(agent, monkeypatch, url: str, *extra_routes) -> None:
    service = agent.llm_service
    monkeypatch.setattr(service, "config", {
        **(service.config or {}),
        "route_priority": ["ollama:local"],
        DISCOVERY_ROUTE_DEADLINE_KEY: SHORT_DEADLINE,
    })
    monkeypatch.setattr(service, "providers", [_silent_ollama_route(service, url), *extra_routes])
    monkeypatch.setattr(service, "_mandate_preference", {})
    service.clear_model_cache()
    service.clear_embedding_discovery_cache()


def _ollama_failure() -> str:
    return _failures()["ollama"]


_OLLAMA_OVERRAN = re.compile(
    r"DiscoveryStepTimeout: chat catalog on ollama:local "
    rf"(did not finish within {SHORT_DEADLINE:g}s"
    rf"|timed out: Ollama model listing did not answer within {SHORT_DEADLINE:g}s)$"
)


@pytest.mark.asyncio
async def test_model_list_with_a_silent_ollama_lists_the_rest_and_marks_ollama_failed(
    tmp_path, monkeypatch, caplog,
    silent_route,  # noqa: F811 - the imported fixture
):
    """The issue's reproduction: before #3646 this never returned."""
    caplog.set_level(logging.WARNING, logger=DISCOVERY_LOGGER)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    async with booted_agent(
        tmp_path, name="Plover", provenance="test:discovery_route_deadline",
        features={"ModelAgent"},
    ) as agent:
        from kestrel_sovereign.features.security.permissions import PermissionLevel

        await agent.get_feature("SecurityFeature").permission_store.set_permission(
            feature_name="ModelAgent", tool_name="list_models",
            level=PermissionLevel.ALLOW, scope="always", reason="test",
        )
        fixture = {
            "name": "fixture:api", "vendor": "fixture", "route": "api", "model": "auto",
            "adapter": _AnsweringCatalog(), "client": object(), "is_cloud": True,
            "is_local": False, "selection_hints": [], "capabilities": {},
        }
        await _on_a_silent_ollama(agent, monkeypatch, silent_route, fixture)

        reply = await await_within_hang_guard(
            agent.process_input("!model-list"), step="!model-list with a silent Ollama"
        )

    assert reply.startswith("Listed 1 model(s)"), reply
    assert '"id": "fixture-chat"' in reply
    assert _OLLAMA_OVERRAN.match(_ollama_failure()), _ollama_failure()
    # One step overran; the route was not asked again in the same listing.
    warnings = _deadline_warnings(caplog)
    assert len(warnings) == 1 and warnings[0].startswith("ollama: chat catalog on ollama:local")


@pytest.mark.asyncio
async def test_an_auto_turn_on_a_silent_ollama_ends_with_the_routes_failure(
    tmp_path, monkeypatch,
    silent_route,  # noqa: F811 - the imported fixture
):
    """``auto`` that discovery could not resolve fails the route, structurally.

    The turn neither waits on the silent daemon's catalog nor sends it the
    literal ``auto`` to generate with: the route chain refuses the route and
    names the discovery timeout. Before #3646 the turn waited on discovery
    forever.

    The route here is not an embedding route. A turn whose memory retrieval
    embeds through a silent daemon waits on that embedding call first, before
    model resolution; that call is generation, not discovery, and this
    deadline does not cover it.
    """
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    async with booted_agent(
        tmp_path, name="Plover", provenance="test:discovery_route_deadline", features=set(),
    ) as agent:
        await _on_a_silent_ollama(agent, monkeypatch, silent_route)
        agent.llm_service.providers[0]["capabilities"]["supports_embeddings"] = False

        with pytest.raises(LLMServiceError) as failed:
            await await_within_hang_guard(
                agent.process_input("Say hello."), step="a turn on a silent auto route"
            )

    message = str(failed.value)
    assert "Model 'auto' is not in the model catalog for route 'ollama:local'" in message
    assert "chat catalog on ollama:local" in message, message
    assert _OLLAMA_OVERRAN.match(_ollama_failure()), _ollama_failure()
