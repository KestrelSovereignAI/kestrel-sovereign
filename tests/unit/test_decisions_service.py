"""``LLMService.decide`` end to end with fake decision adapters (#3424).

Covers privacy tightening, discovery scoped to privacy-surviving routes, no
re-dispatch after a send, timeout and cancellation, content-free telemetry on
the decision metrics series, metering compatibility, pin canaries, and
discovery replace-on-success / keep-on-failure.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.llm.decisions import (
    ChoiceQuestion,
    DecisionModelInfo,
    DecisionRequest,
    DecisionTimeout,
    DecisionTransportError,
    DecisionUnavailable,
    NoulQuestion,
    UnavailableReason,
)

from kestrel_sovereign.llm import modality_recording as modality_recording_mod
from kestrel_sovereign.llm.decision_service import DecisionServiceMixin
from kestrel_sovereign.llm.decisions.config import (
    DecisionRouteConfig,
    parse_service_decision_config,
)
from kestrel_sovereign.llm.decisions.http import DecisionHTTPError
from kestrel_sovereign.llm.decisions.resolve import PinStatus, RouteDecisionState
from kestrel_sovereign.llm.decisions.thresholds import ThresholdBook
from kestrel_sovereign.llm.model_discovery import ModelDiscoveryMixin
from kestrel_sovereign.llm.service import LLMService

SECRET = "the patient said something private"


def _request() -> DecisionRequest:
    return DecisionRequest(
        state={"message": SECRET},
        questions={
            "taken": NoulQuestion(instructions="Did the patient take the medication?"),
            "topic": ChoiceQuestion(instructions="Topic?", options={"meds": None, "other": None}),
        },
    )


def _answer_body(**usage: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "answers": {
            "taken": {"type": "noul", "noul": 0.02},
            "topic": {"type": "choice", "choice": "meds",
                      "probabilities": {"meds": 0.9, "other": 0.1}, "confidence": 0.8},
        }
    }
    if usage:
        body["usage"] = usage
    return body


class FakeDecisionAdapter:
    def __init__(self, models: List[str], *, body=None, error: Optional[BaseException] = None,
                 delay: float = 0.0, discovery_error: Optional[BaseException] = None):
        self.models = models
        self.body = body if body is not None else _answer_body(input_tokens=42, cost=0.00002)
        self.error = error
        self.delay = delay
        self.discovery_error = discovery_error
        self.decide_calls: List[str] = []
        self.discovery_calls = 0

    async def list_decision_models(self, client):
        self.discovery_calls += 1
        if self.discovery_error is not None:
            raise self.discovery_error
        return [DecisionModelInfo(id=m, vendor="", route="", context_limit=8192) for m in self.models]

    async def adecide(self, client, model, request, *, timeout):
        self.decide_calls.append(model)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.body


def _route(name: str, adapter: FakeDecisionAdapter, *, local: bool, pin: Optional[str] = None):
    vendor, _, route = name.partition(":")
    return {
        "name": name, "vendor": vendor, "route": route, "is_local": local, "is_cloud": not local,
        "client": object(), "adapter": adapter, "model": "auto",
        "capabilities": {"supports_decisions": True},
        "decision_state": RouteDecisionState(config=DecisionRouteConfig(pin=pin)),
    }


def _service(routes, *, live_local_only: bool = False, thresholds=None, llm_cfg=None) -> LLMService:
    service = LLMService.__new__(LLMService)
    service.providers = routes
    service._disabled_routes = set()
    service._force_local_only_provider = lambda: live_local_only
    service._decision_config = parse_service_decision_config(llm_cfg or {})
    service._decision_thresholds = ThresholdBook.from_config(thresholds or {})
    service._observability_store = MagicMock()
    service._observability_store.log_llm_call = AsyncMock()
    service._owner_agent_did = "did"
    service._track_model_usage = AsyncMock()
    return service


def _logged(service) -> List[Dict[str, Any]]:
    return [call.kwargs for call in service._observability_store.log_llm_call.await_args_list]


@pytest.mark.asyncio
async def test_decide_returns_normalised_answers_and_records_without_content() -> None:
    adapter = FakeDecisionAdapter(["nimble"])
    service = _service([_route("ollama:local", adapter, local=True)])

    result = await service.decide(_request(), caller="medication", timeout_seconds=5)

    assert result.answers["taken"].p_true == pytest.approx(0.02)
    assert result.answers["topic"].choice == "meds"
    assert (result.vendor, result.route, result.model) == ("ollama", "ollama:local", "nimble")
    assert result.calibrated is None and dict(result.thresholds) == {}
    assert result.input_tokens == 42

    [row] = _logged(service)
    assert row["provider"] == "ollama:local" and row["model"] == "nimble" and row["success"]
    assert row["metadata"]["modality"] == "decision" and row["metadata"]["caller"] == "medication"
    flattened = repr(row)
    assert SECRET not in flattened and "Did the patient" not in flattened
    assert row["user_prompt"] is None and row["response"] is None
    service._track_model_usage.assert_awaited_once_with("nimble", "ollama:local", tokens=42)


@pytest.mark.asyncio
async def test_privacy_only_tightens_and_cloud_is_never_contacted() -> None:
    cloud = FakeDecisionAdapter(["jev"])
    local = FakeDecisionAdapter(["nimble"])
    service = _service(
        [_route("openrouter:api", cloud, local=False), _route("ollama:local", local, local=True)],
        live_local_only=True,
    )

    # A caller passing local_only=False cannot loosen the live restriction.
    result = await service.decide(_request(), caller="c", timeout_seconds=5, local_only=False)

    assert result.route == "ollama:local"
    assert cloud.discovery_calls == 0 and cloud.decide_calls == []


@pytest.mark.asyncio
async def test_caller_can_tighten_privacy() -> None:
    cloud = FakeDecisionAdapter(["jev"])
    service = _service([_route("openrouter:api", cloud, local=False)])
    with pytest.raises(DecisionUnavailable) as excinfo:
        await service.decide(_request(), caller="c", timeout_seconds=5, local_only=True)
    assert excinfo.value.reason is UnavailableReason.NO_LOCAL_ROUTE
    assert cloud.discovery_calls == 0


@pytest.mark.asyncio
async def test_a_failed_dispatch_is_never_resent_to_another_route() -> None:
    failing = FakeDecisionAdapter(["jev"], error=DecisionTransportError("openrouter: HTTP 503"))
    standby = FakeDecisionAdapter(["nimble"])
    service = _service(
        [_route("openrouter:api", failing, local=False), _route("ollama:local", standby, local=True)]
    )
    with pytest.raises(DecisionTransportError):
        await service.decide(_request(), caller="c", timeout_seconds=5)
    assert failing.decide_calls == ["jev"] and standby.decide_calls == []
    [row] = _logged(service)
    assert row["success"] is False and row["error_message"] == "DecisionTransportError"


@pytest.mark.asyncio
async def test_timeout_is_typed_and_still_recorded() -> None:
    slow = FakeDecisionAdapter(["nimble"], delay=5)
    service = _service([_route("ollama:local", slow, local=True)])
    with pytest.raises(DecisionTimeout):
        await service.decide(_request(), caller="c", timeout_seconds=0.05)
    [row] = _logged(service)
    assert row["success"] is False and row["metadata"]["usage_available"] is False


@pytest.mark.asyncio
async def test_cancellation_propagates_and_the_record_survives() -> None:
    slow = FakeDecisionAdapter(["nimble"], delay=5)
    service = _service([_route("ollama:local", slow, local=True)])
    task = asyncio.create_task(service.decide(_request(), caller="c", timeout_seconds=30))
    while not slow.decide_calls:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await service.drain_modality_records()
    [row] = _logged(service)
    assert row["success"] is False and row["error_message"] == "CancelledError"


@pytest.mark.asyncio
async def test_a_slow_record_does_not_hold_the_caller(monkeypatch) -> None:
    monkeypatch.setattr(modality_recording_mod, "USAGE_RECORD_TIMEOUT", 0.01)
    adapter = FakeDecisionAdapter(["nimble"])
    service = _service([_route("ollama:local", adapter, local=True)])
    release = asyncio.Event()

    async def slow_log(**kwargs):
        await release.wait()

    service._observability_store.log_llm_call = AsyncMock(side_effect=slow_log)
    result = await service.decide(_request(), caller="c", timeout_seconds=5)
    assert result.model == "nimble"
    assert service._pending_modality_records()
    release.set()
    await service.drain_modality_records()
    assert not service._pending_modality_records()


@pytest.mark.asyncio
async def test_unavailable_sends_nothing_and_records_nothing() -> None:
    adapter = FakeDecisionAdapter(["a", "b"])  # ambiguous, no hints
    service = _service([_route("openrouter:api", adapter, local=False)])
    with pytest.raises(DecisionUnavailable) as excinfo:
        await service.decide(_request(), caller="c", timeout_seconds=5)
    assert excinfo.value.reason is UnavailableReason.NO_CANDIDATE
    assert adapter.decide_calls == [] and _logged(service) == []


@pytest.mark.asyncio
async def test_thresholds_are_resolved_for_the_answering_model() -> None:
    adapter = FakeDecisionAdapter(["nimble"])
    service = _service(
        [_route("ollama:local", adapter, local=True)],
        thresholds={"thresholds": {"medication": {
            "uncalibrated": "default", "default": {"taken": 0.5, "topic": 0.5},
            "models": {"ollama:local/nimble": {"taken": 0.3, "topic": 0.6}},
        }}},
    )
    result = await service.decide(_request(), caller="medication", timeout_seconds=5)
    assert result.calibrated is True and dict(result.thresholds) == {"taken": 0.3, "topic": 0.6}
    assert _logged(service)[0]["metadata"]["calibrated"] is True


# ---------------------------------------------------------------------------
# Discovery and pin canaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discovery_replaces_on_success_and_keeps_on_failure() -> None:
    adapter = FakeDecisionAdapter(["nimble", "tev1"])
    route = _route("ollama:local", adapter, local=True)
    service = _service([route])

    await service.reconcile_decision_capabilities(use_cache=False)
    assert [m.id for m in route["decision_state"].models] == ["nimble", "tev1"]
    assert route["decision_state"].models[0].route == "ollama:local"

    adapter.models = ["nimble"]
    await service.reconcile_decision_capabilities(use_cache=False)
    assert [m.id for m in route["decision_state"].models] == ["nimble"]

    adapter.discovery_error = RuntimeError("daemon down")
    await service.reconcile_decision_capabilities(use_cache=False)
    assert [m.id for m in route["decision_state"].models] == ["nimble"]
    assert route["decision_state"].discovery_stale_since is not None

    calls = adapter.discovery_calls
    await service.reconcile_decision_capabilities(use_cache=True)
    assert adapter.discovery_calls == calls  # cache hit: no network


@pytest.mark.asyncio
async def test_pin_canary_outcomes() -> None:
    adapter = FakeDecisionAdapter([], body={"answers": {"canary": {"type": "noul", "noul": 0.9}}})
    route = _route("ollama:local", adapter, local=True, pin="nimble")
    service = _service([route])
    state = route["decision_state"]

    await service.reconcile_decision_capabilities(use_cache=False)
    assert state.pin_status is PinStatus.VERIFIED and state.pin_info.id == "nimble"
    assert _logged(service)[-1]["metadata"]["caller"] == "kestrel.canary"

    # A transient failure leaves a verified pin verified.
    adapter.error = DecisionTransportError("ollama: connection refused")
    await service.reconcile_decision_capabilities(use_cache=False)
    assert state.pin_status is PinStatus.VERIFIED and state.canary_stale_since is not None

    # A definitive 404 flips it.
    adapter.error = DecisionHTTPError("ollama: HTTP 404", status_code=404)
    await service.reconcile_decision_capabilities(use_cache=False)
    assert state.pin_status is PinStatus.UNVERIFIED and state.pin_info is None


@pytest.mark.asyncio
async def test_pin_canary_with_a_malformed_answer_is_unverified() -> None:
    adapter = FakeDecisionAdapter([], body={"answers": {}})
    route = _route("ollama:local", adapter, local=True, pin="nimble")
    service = _service([route])
    await service.reconcile_decision_capabilities(use_cache=False)
    assert route["decision_state"].pin_status is PinStatus.UNVERIFIED


@pytest.mark.asyncio
async def test_cold_routes_are_discovered_inside_decide_only_after_privacy() -> None:
    cloud = FakeDecisionAdapter(["jev"])
    local = FakeDecisionAdapter(["nimble"])
    service = _service(
        [_route("openrouter:api", cloud, local=False), _route("ollama:local", local, local=True)]
    )
    await service.decide(_request(), caller="c", timeout_seconds=5, local_only=True)
    assert local.discovery_calls == 1 and cloud.discovery_calls == 0


def test_llm_service_uses_the_real_decision_reconcile() -> None:
    assert (
        LLMService.reconcile_decision_capabilities
        is DecisionServiceMixin.reconcile_decision_capabilities
    )
    assert (
        LLMService.reconcile_decision_capabilities
        is not ModelDiscoveryMixin.reconcile_decision_capabilities
    )


# ---------------------------------------------------------------------------
# Recorder: metrics series and metering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decisions_count_only_in_the_decision_metrics_series() -> None:
    from kestrel_sdk import metrics

    if not metrics.PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus-client not installed")
    adapter = FakeDecisionAdapter(["nimble-metrics"])
    service = _service([_route("ollama:local", adapter, local=True)])

    def chat_count() -> float:
        return metrics.REGISTRY.get_sample_value(
            "kestrel_llm_calls_total",
            {"provider": "ollama:local", "model": "nimble-metrics", "success": "True"},
        ) or 0.0

    def decision_count() -> float:
        return metrics.REGISTRY.get_sample_value(
            "kestrel_llm_decision_calls_total",
            {"provider": "ollama:local", "model": "nimble-metrics", "caller": "m", "success": "True"},
        ) or 0.0

    before_chat, before_decision = chat_count(), decision_count()
    await service.decide(_request(), caller="m", timeout_seconds=5)
    assert chat_count() == before_chat
    assert decision_count() == before_decision + 1


@pytest.mark.asyncio
async def test_original_signature_metering_callback_still_bills_decisions() -> None:
    adapter = FakeDecisionAdapter(["nimble"])
    service = _service([_route("ollama:local", adapter, local=True)])
    billed: List[Dict[str, Any]] = []

    async def original(*, companion_id, user_id, provider, model, prompt_tokens, completion_tokens):
        billed.append(dict(provider=provider, model=model, prompt_tokens=prompt_tokens))

    service.set_metering_callback(original)
    from kestrel_sovereign.llm.invocation_context import LLMInvocationContext

    service._resolve_invocation_context = lambda *a, **k: LLMInvocationContext(
        session_id="s", companion_id="comp", user_id="user"
    )
    await service.decide(_request(), caller="c", timeout_seconds=5)
    assert billed == [dict(provider="ollama:local", model="nimble", prompt_tokens=42)]


@pytest.mark.asyncio
async def test_metering_callback_that_names_modality_receives_it() -> None:
    adapter = FakeDecisionAdapter(["nimble"])
    service = _service([_route("ollama:local", adapter, local=True)])
    seen: List[str] = []

    async def aware(*, companion_id, user_id, provider, model, prompt_tokens,
                    completion_tokens, modality):
        seen.append(modality)

    service.set_metering_callback(aware)
    from kestrel_sovereign.llm.invocation_context import LLMInvocationContext

    service._resolve_invocation_context = lambda *a, **k: LLMInvocationContext(
        session_id="s", companion_id="comp", user_id="user"
    )
    await service.decide(_request(), caller="c", timeout_seconds=5)
    assert seen == ["decision"]


# ---------------------------------------------------------------------------
# Threshold keys and caller defaults (slice 3)
# ---------------------------------------------------------------------------


def _many_candidates_request(n: int = 3) -> DecisionRequest:
    return DecisionRequest(
        state={"question": "pet name?", "candidates": {f"c{i}": f"text {i}" for i in range(n)}},
        questions={f"c{i}": NoulQuestion(instructions=f"Does `candidates.c{i}` answer `question`?")
                   for i in range(n)},
    )


class _NoulAdapter(FakeDecisionAdapter):
    def __init__(self, models, n):
        super().__init__(models, body={"answers": {f"c{i}": {"type": "noul", "noul": 0.5}
                                                   for i in range(n)}})


@pytest.mark.asyncio
async def test_shared_threshold_key_covers_many_questions() -> None:
    adapter = _NoulAdapter(["nimble"], 3)
    service = _service(
        [_route("ollama:local", adapter, local=True)],
        thresholds={"thresholds": {"gate": {
            "uncalibrated": "refuse",
            "models": {"ollama:local/nimble": {"answers": 0.7}},
        }}},
    )
    keys = {f"c{i}": "answers" for i in range(3)}
    result = await service.decide(_many_candidates_request(), caller="gate",
                                  timeout_seconds=5, threshold_keys=keys)
    assert result.calibrated is True
    assert dict(result.thresholds) == {"c0": 0.7, "c1": 0.7, "c2": 0.7}


@pytest.mark.asyncio
async def test_caller_defaults_apply_until_config_overrides_them() -> None:
    adapter = _NoulAdapter(["nimble"], 2)
    keys = {"c0": "answers", "c1": "answers"}

    service = _service([_route("ollama:local", adapter, local=True)])
    result = await service.decide(_many_candidates_request(2), caller="gate", timeout_seconds=5,
                                  threshold_keys=keys, default_thresholds={"answers": 0.5})
    assert result.calibrated is False and dict(result.thresholds) == {"c0": 0.5, "c1": 0.5}

    service = _service([_route("ollama:local", adapter, local=True)],
                       thresholds={"thresholds": {"gate": {"default": {"answers": 0.65}}}})
    result = await service.decide(_many_candidates_request(2), caller="gate", timeout_seconds=5,
                                  threshold_keys=keys, default_thresholds={"answers": 0.5})
    assert dict(result.thresholds) == {"c0": 0.65, "c1": 0.65}


@pytest.mark.asyncio
async def test_threshold_key_and_default_validation() -> None:
    from kestrel_sdk.llm.decisions import DecisionRequestInvalid

    adapter = _NoulAdapter(["nimble"], 2)
    service = _service([_route("ollama:local", adapter, local=True)])
    with pytest.raises(DecisionRequestInvalid, match="unknown question"):
        await service.decide(_many_candidates_request(2), caller="gate", timeout_seconds=5,
                             threshold_keys={"c9": "answers"})
    with pytest.raises(DecisionRequestInvalid, match="not a valid id"):
        await service.decide(_many_candidates_request(2), caller="gate", timeout_seconds=5,
                             threshold_keys={"c0": "has space"})
    with pytest.raises(DecisionRequestInvalid, match="in \\[0, 1\\]"):
        await service.decide(_many_candidates_request(2), caller="gate", timeout_seconds=5,
                             default_thresholds={"c0": 1.5})
    with pytest.raises(DecisionRequestInvalid, match="no threshold"):
        await service.decide(_many_candidates_request(2), caller="gate", timeout_seconds=5,
                             threshold_keys={"c0": "answers", "c1": "answers"},
                             default_thresholds={"other": 0.5})
    assert adapter.decide_calls == []
