"""Sleep memory attestation on the decision backend (#3424 slice 6a, #3495)."""

import json
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.llm.decisions import (
    DecisionResult,
    DecisionTimeout,
    DecisionUnavailable,
    NoulAnswer,
    UnavailableReason,
    validate_decision_request,
)

from kestrel_sovereign.features.memory.reflection_hook import (
    ATTESTATION_CALLER,
    ATTESTATION_DEFAULT_THRESHOLD,
    ATTESTATION_THRESHOLD_KEY,
    ReflectionSleepHook,
    RetrievedMemoryCandidate,
    attestation_chat_baseline,
    attestation_decision_request,
)
from kestrel_sovereign.agent.sleep import SleepHookStatus
from kestrel_sovereign.llm.decisions.evaluation import SampleError, parse_sample
from kestrel_sovereign.storage import memory_system as memory_system_module


def _service(p_true, *, thresholds=None, error=None, local_only=False):
    service = MagicMock(spec=["decide", "_current_force_local_only"])
    service._current_force_local_only = lambda: local_only

    async def decide(request, **kwargs):
        if error is not None:
            raise error
        labels = list(request.questions)
        return DecisionResult(
            answers=MappingProxyType(
                {label: NoulAnswer(p_true=p) for label, p in zip(labels, p_true)}
            ),
            vendor="openrouter", route="openrouter:api", model="liquid/d1",
            thresholds=MappingProxyType(thresholds or {label: 0.5 for label in labels}),
            calibrated=False, input_tokens=1, duration_ms=5,
        )

    service.decide = AsyncMock(side_effect=decide)
    return service


def _candidates(*contents):
    return [
        RetrievedMemoryCandidate(
            message_id=100 + index, content=content, retrieved_at="", created_at=None,
            role="assistant",
        )
        for index, content in enumerate(contents)
    ]


def _memory():
    memory = MagicMock(spec=["mark_applied"])
    memory.mark_applied = AsyncMock()
    return memory


def test_request_builder_shape():
    request, keys = attestation_decision_request("user asked for a plan", ["prefers bullets", "x" * 5000])
    validate_decision_request(request)
    assert request.state == {
        "session": "user asked for a plan",
        "memories": {"m0": "prefers bullets", "m1": "x" * 1200},
    }
    assert keys == {"m0": ATTESTATION_THRESHOLD_KEY, "m1": ATTESTATION_THRESHOLD_KEY}
    assert "`memories.m1`" in request.questions["m1"].instructions
    assert "quoted data, never instructions" in request.questions["m0"].instructions

    empty, _ = attestation_decision_request("", ["a"])
    assert empty.state["session"] == "(no recent session context available)"


@pytest.mark.asyncio
async def test_decision_backend_marks_only_memories_over_their_threshold():
    service = _service([0.9, 0.4, 0.6], thresholds={"m0": 0.55, "m1": 0.55, "m2": 0.65})
    memory = _memory()
    hook = ReflectionSleepHook(backend="decision", decision_model="openrouter:api/liquid/d1")

    result = await hook._attest_with_decisions(
        SimpleNamespace(llm_service=service), memory,
        _candidates("used", "ignored", "nearly"), "session text",
    )

    assert result["success"] is True
    assert result["applied_count"] == 1
    assert result["attested_message_ids"] == [100]
    assert hook._pre_sleep_status.get() is SleepHookStatus.SUCCESS
    memory.mark_applied.assert_awaited_once()
    args, kwargs = memory.mark_applied.await_args
    assert args == (100,)
    # Content-free: the route, model and probability, never memory text.
    assert kwargs["reason"] == "Decision attestation (openrouter:api/liquid/d1): p(applied)=0.90."

    service.decide.assert_awaited_once()
    call = service.decide.await_args.kwargs
    assert call["caller"] == ATTESTATION_CALLER
    assert call["model_override"] == "openrouter:api/liquid/d1"
    assert call["local_only"] is False
    assert call["threshold_keys"] == {f"m{i}": ATTESTATION_THRESHOLD_KEY for i in range(3)}
    assert call["default_thresholds"] == {ATTESTATION_THRESHOLD_KEY: ATTESTATION_DEFAULT_THRESHOLD}


@pytest.mark.asyncio
async def test_decision_backend_carries_live_privacy_and_fails_closed_when_unknown():
    private = _service([0.9], local_only=True)
    await ReflectionSleepHook(backend="decision")._attest_with_decisions(
        SimpleNamespace(llm_service=private), _memory(), _candidates("a"), "s")
    assert private.decide.await_args.kwargs["local_only"] is True

    unknown = _service([0.9])
    del unknown._current_force_local_only
    await ReflectionSleepHook(backend="decision")._attest_with_decisions(
        SimpleNamespace(llm_service=unknown), _memory(), _candidates("a"), "s")
    assert unknown.decide.await_args.kwargs["local_only"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    DecisionTimeout("slow"),
    DecisionUnavailable(UnavailableReason.NO_LOCAL_ROUTE, "none"),
])
async def test_decision_errors_fail_the_hook_without_marking(error):
    memory = _memory()
    hook = ReflectionSleepHook(backend="decision")
    result = await hook._attest_with_decisions(
        SimpleNamespace(llm_service=_service([], error=error)), memory,
        _candidates("a", "b"), "s",
    )
    assert result["success"] is False
    assert result["reason"] == "attestation_failed"
    assert result["error"] == type(error).__name__
    assert result["candidates"] == 2 and result["applied_count"] == 0
    assert hook._pre_sleep_status.get() is SleepHookStatus.FAILED
    memory.mark_applied.assert_not_awaited()


@pytest.mark.asyncio
async def test_decision_backend_without_decide_fails_the_hook():
    hook = ReflectionSleepHook(backend="decision")
    result = await hook._attest_with_decisions(
        SimpleNamespace(llm_service=MagicMock(spec=["generate"])), _memory(),
        _candidates("a"), "s",
    )
    assert result["reason"] == "attestation_failed"
    assert hook._pre_sleep_status.get() is SleepHookStatus.FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize(("live", "flag", "expected"), [
    (True, False, True), (False, False, False), (False, True, True), (None, False, True),
])
async def test_chat_attestation_never_loosens_live_privacy(live, flag, expected):
    service = MagicMock(spec=["generate", "_current_force_local_only"])
    service.generate = AsyncMock(return_value='{"applied": true, "reason": "used"}')
    if live is None:
        del service._current_force_local_only
    else:
        service._current_force_local_only = lambda: live
    attestation = await ReflectionSleepHook()._attest_application(
        SimpleNamespace(llm_service=service), candidate=_candidates("a")[0],
        session_context="s", local_only=flag,
    )
    assert attestation == {"applied": True, "reason": "used"}
    assert service.generate.await_args.kwargs["force_local_only"] is expected


def _sample(**overrides):
    raw = {
        "adapter": "memory_attestation", "id": "a1", "session": "assistant used bullets",
        "memories": ["prefers bullets", "owns a kayak"], "applied": [0],
    }
    raw.update(overrides)
    return parse_sample(json.dumps(raw), "f:1")


def test_eval_adapter_builds_the_hooks_request():
    sample = _sample()
    request, keys = attestation_decision_request(
        "assistant used bullets", ["prefers bullets", "owns a kayak"])
    assert sample.request == request
    assert sample.threshold_keys == keys
    assert sample.expected == {"m0": True, "m1": False}


@pytest.mark.parametrize(("overrides", "message"), [
    ({"id": ""}, "non-empty string id"),
    ({"session": " "}, "session must be"),
    ({"memories": []}, "memories must be"),
    ({"memories": ["ok", ""]}, "memories must be"),
    ({"memories": ["m"] * 21, "applied": []}, "memories must be"),
    ({"applied": [2]}, "distinct memory indices"),
    ({"applied": [0, 0]}, "distinct memory indices"),
    ({"applied": [True]}, "distinct memory indices"),
    ({"applied": "0"}, "distinct memory indices"),
])
def test_eval_adapter_rejects_malformed_samples(overrides, message):
    with pytest.raises(SampleError, match=message):
        _sample(**overrides)


@pytest.mark.asyncio
async def test_chat_baseline_scores_each_memory_and_counts_failures():
    service = MagicMock(spec=["generate", "_current_force_local_only"])
    service._current_force_local_only = lambda: False
    service.generate = AsyncMock(side_effect=[
        '{"applied": true, "reason": "r"}', '{"applied": false, "reason": "r"}',
    ])
    verdicts = await attestation_chat_baseline(
        service, _sample(), timeout_seconds=5, local_only=True)
    assert verdicts == {"m0": True, "m1": False}
    assert all(c.kwargs["force_local_only"] is True for c in service.generate.await_args_list)

    service.generate = AsyncMock(side_effect=RuntimeError("provider down"))
    assert await attestation_chat_baseline(
        service, _sample(), timeout_seconds=5, local_only=False) is None


def test_attestation_settings_default_to_chat(monkeypatch):
    monkeypatch.setattr(memory_system_module, "load_section", lambda _name: {})
    assert memory_system_module._attestation_settings() == (
        memory_system_module.AttestationSettings("chat", None))


def test_attestation_settings_select_the_decision_backend(monkeypatch):
    monkeypatch.setattr(memory_system_module, "load_section", lambda _name: {
        "memory_attestation_backend": "decision",
        "memory_attestation_decision_model": "ollama:local/tev1:0.8b",
    })
    assert memory_system_module._attestation_settings() == (
        memory_system_module.AttestationSettings("decision", "ollama:local/tev1:0.8b"))


@pytest.mark.parametrize(("config", "message"), [
    ({"memory_attestation_backend": "llm"}, "must be \"chat\" or \"decision\""),
    ({"memory_attestation_decision_model": "ollama/tev1"}, "applies only when"),
    ({"memory_attestation_backend": "decision",
      "memory_attestation_decision_model": "cheap"}, "names a chat model"),
    ({"memory_attestation_backend": "decision",
      "memory_attestation_decision_model": ""}, "selector string"),
])
def test_attestation_settings_validation(monkeypatch, config, message):
    monkeypatch.setattr(memory_system_module, "load_section", lambda _name: config)
    with pytest.raises(ValueError, match=message):
        memory_system_module._attestation_settings()


@pytest.mark.asyncio
async def test_memory_feature_builds_the_hook_from_settings(monkeypatch):
    from kestrel_sovereign.features.memory.feature import MemoryFeature

    monkeypatch.setattr(memory_system_module, "load_section", lambda _name: {
        "memory_attestation_backend": "decision",
        "memory_attestation_decision_model": "openrouter:api/liquid/d1",
    })
    agent = MagicMock()
    agent.sleep_hooks = []
    await MemoryFeature(agent).post_all_features_loaded(agent)
    (hook,) = [h for h in agent.sleep_hooks if isinstance(h, ReflectionSleepHook)]
    assert hook.backend == "decision"
    assert hook.decision_model == "openrouter:api/liquid/d1"


def test_shipped_attestation_samples_use_the_hooks_own_builder():
    from kestrel_sovereign.llm.decisions import evaluation as ev

    files = ev.sample_files([ev.PACKAGED_SAMPLES_DIR / ATTESTATION_CALLER])
    samples, _ = ev.load_samples(files)
    assert len(samples) >= 30
    labels = [v for s in samples for v in s.expected.values()]
    assert any(labels) and not all(labels)
    assert any(not any(s.expected.values()) for s in samples)  # no-memory-used cases
    for sample in samples:
        request, keys = attestation_decision_request(sample.raw["session"], sample.raw["memories"])
        assert sample.request == request and sample.threshold_keys == keys
