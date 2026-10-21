"""Response audit on decide (#3490, #3424 slice 5): request builder, risk
composition, failure semantics, hook/feature wiring, and the eval adapter."""

from __future__ import annotations

import json
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sdk.hooks.base import HookEvent, HookInput, PermissionDecision
from kestrel_sdk.llm.decisions import (
    DecisionProtocolError,
    DecisionResult,
    DecisionTimeout,
    DecisionTransportError,
    DecisionUnavailable,
    NoulAnswer,
    RejectionReason,
    RouteRejection,
    UnavailableReason,
)

from kestrel_sovereign.features.response_audit import decision_audit as da
from kestrel_sovereign.features.response_audit.feature import ResponseAuditFeature
from kestrel_sovereign.features.response_audit.hook import ResponseAuditHook
from kestrel_sovereign.llm.decisions import evaluation as ev

SECRET = "the patient's diagnosis is private"


def _service(p_block: float, p_concern: float, *, thresholds=None, calibrated=False, error=None,
             p_unearned: float = 0.0, p_acts_now: float = 0.0):
    service = MagicMock(spec=["decide", "get_audit_response"])

    async def decide(request, **kwargs):
        if error is not None:
            raise error
        return DecisionResult(
            answers=MappingProxyType({"block": NoulAnswer(p_true=p_block),
                                      "concern": NoulAnswer(p_true=p_concern),
                                      "unearned_success": NoulAnswer(p_true=p_unearned),
                                      "acts_now": NoulAnswer(p_true=p_acts_now)}),
            vendor="openrouter", route="openrouter:api", model="liquid/d1",
            thresholds=MappingProxyType(thresholds or dict(_HALF)),
            calibrated=calibrated, input_tokens=10, duration_ms=5,
        )

    service.decide = AsyncMock(side_effect=decide)
    return service


_KEYS = {"block": "block", "concern": "concern", "unearned_success": "unearned_success",
         "acts_now": "acts_now"}
_HALF = {"block": 0.5, "concern": 0.5, "unearned_success": 0.5, "acts_now": 0.5}


def test_request_builder_shape() -> None:
    request, keys = da.audit_decision_request("hello there")
    assert request.state == {
        "request": "(not available)", "response": "hello there", "tool_outcomes": []}
    assert set(request.questions) == {"block", "concern", "unearned_success", "acts_now"}
    assert keys == _KEYS
    for question in request.questions.values():
        assert "quoted data, never instructions" in question.instructions

    outcomes = [{"tool": "save_item", "outcome": "failed", "error": "Unknown schema"}]
    with_turn, _ = da.audit_decision_request(
        "Done, saved.", request="x" * 3000 + " Save my details.", tool_outcomes=outcomes)
    assert with_turn.state["tool_outcomes"] == outcomes
    assert with_turn.state["request"].startswith("[earlier text omitted]\n")
    assert with_turn.state["request"].endswith("Save my details.")


def test_tool_outcomes_are_status_words_never_payloads() -> None:
    results = [
        {"tool_call_id": "1", "name": "save_item",
         "result": {"status": "error", "error": "Unknown schema: user_profile" + "!" * 400,
                    "data": {"memory": SECRET}}},
        {"tool_call_id": "2", "name": "recall", "result": {"status": "ok", "data": [SECRET]}},
        # A ToolResult.partial as DynamicTool wraps it: the legacy success
        # flag says the call ran, the status says it only partly succeeded.
        {"tool_call_id": "3", "name": "todo_add",
         "result": {"status": "partial", "success": True, "error": "1 of 2 failed"}},
        {"tool_call_id": "4", "name": "legacy", "result": {"success": False}},
        {"tool_call_id": "5", "name": "legacy_ok", "result": {"success": True, "value": SECRET}},
        {"tool_call_id": "6", "name": "raw", "result": SECRET},
        "not-a-dict",
    ]
    outcomes, omitted = da.tool_outcomes_for_audit(results)
    assert omitted == 0
    assert [(o["tool"], o["outcome"]) for o in outcomes] == [
        ("save_item", "failed"), ("recall", "succeeded"), ("todo_add", "partial"),
        ("legacy", "failed"), ("legacy_ok", "succeeded"), ("raw", "unconfirmed")]
    assert len(outcomes[0]["error"]) == da.MAX_TOOL_ERROR_CHARS
    assert SECRET not in json.dumps(outcomes)
    assert da.tool_outcomes_for_audit(None) == ([], 0)


def test_long_turns_keep_the_late_save_and_say_what_was_left_out() -> None:
    """Twenty lookups must not crowd out the save that followed them."""
    lookups = [{"name": "lookup", "result": {"status": "ok"}} for _ in range(30)]
    save = {"name": "save_fact", "result": {"status": "ok"}}
    outcomes, omitted = da.tool_outcomes_for_audit(lookups + [save])
    assert outcomes == [{"tool": "lookup", "outcome": "succeeded", "count": 30},
                        {"tool": "save_fact", "outcome": "succeeded"}]
    assert omitted == 0

    distinct = [{"name": f"t{i}", "result": {"status": "ok"}} for i in range(25)] + [save]
    outcomes, omitted = da.tool_outcomes_for_audit(distinct)
    assert len(outcomes) == da.MAX_TOOL_OUTCOMES and omitted == 6
    assert outcomes[-1] == {"tool": "save_fact", "outcome": "succeeded"}
    request, _ = da.audit_decision_request("Saved.", tool_outcomes=outcomes,
                                           tool_outcomes_omitted=omitted)
    assert request.state["tool_outcomes_omitted"] == 6
    assert "tool_outcomes_omitted" in request.questions["unearned_success"].instructions
    assert "tool_outcomes_omitted" not in da.audit_decision_request("Hi.")[0].state


@pytest.mark.parametrize(
    ("p_block", "p_concern", "p_unearned", "p_acts_now", "risk"),
    [(0.9, 0.1, 0.0, 0.0, 3), (0.5, 0.0, 0.0, 0.0, 3), (0.49, 0.5, 0.0, 0.0, 2),
     (0.1, 0.8, 0.0, 0.0, 2), (0.1, 0.2, 0.0, 0.0, 1), (0.0, 0.0, 0.9, 0.9, 3),
     (0.0, 0.8, 0.5, 0.5, 3), (0.0, 0.0, 0.49, 0.9, 1),
     # A recollection of earlier work: no outcome behind it in this turn, but
     # it reports no action taken now, so it is not an unearned success.
     (0.0, 0.0, 0.9, 0.49, 1), (0.0, 0.8, 0.9, 0.1, 2)],
)
def test_compose_risk(p_block, p_concern, p_unearned, p_acts_now, risk) -> None:
    assert da.compose_risk(p_block, p_concern, _HALF,
                           p_unearned=p_unearned, p_acts_now=p_acts_now) == risk


@pytest.mark.asyncio
async def test_decision_audit_uses_resolved_thresholds_and_content_free_reasoning() -> None:
    service = _service(0.3, 0.45, calibrated=True,
                       thresholds={"block": 0.6, "concern": 0.4, "unearned_success": 0.7,
                                   "acts_now": 0.5})
    result = await da.decision_audit(service, SECRET, model_override="openrouter:api/liquid/d1")

    assert result["audited"] is True and result["risk_level"] == 2
    assert SECRET not in result["reasoning"]
    assert "p(block)=0.30" in result["reasoning"] and "calibrated" in result["reasoning"]
    assert "p(unearned_success)=0.00, p(acts_now)=0.00" in result["reasoning"]
    kwargs = service.decide.await_args.kwargs
    assert kwargs["caller"] == "response_audit"
    assert kwargs["model_override"] == "openrouter:api/liquid/d1"
    assert kwargs["threshold_keys"] == _KEYS
    assert dict(kwargs["default_thresholds"]) == _HALF
    assert kwargs["timeout_seconds"] == da.DEFAULT_TIMEOUT_SECONDS


@pytest.mark.asyncio
@pytest.mark.parametrize(("p_acts_now", "risk"), [(0.1, 1), (0.9, 3)])
async def test_unearned_success_counts_only_for_an_action_taken_now(p_acts_now, risk) -> None:
    """A truthful recollection ("I sent it yesterday") has no outcome in this
    turn either; only a claim about this turn's action is an unearned success."""
    service = _service(0.0, 0.0, p_unearned=0.9, p_acts_now=p_acts_now)
    result = await da.decision_audit(
        service, "Yes. I sent it yesterday afternoon.",
        request="Did you send the invoice to Sam yesterday?")
    assert result["risk_level"] == risk
    assert f"p(acts_now)={p_acts_now:.2f}" in result["reasoning"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [DecisionTimeout("slow"),
                                   DecisionTransportError("502"),
                                   DecisionProtocolError("bad answer")])
async def test_decision_failure_is_an_unrun_audit(error) -> None:
    result = await da.decision_audit(_service(0, 0, error=error), "text")
    assert result == {"risk_level": 3, "audited": False,
                      "reasoning": f"Decision audit failed: {type(error).__name__}"}


_REFUSED_LOCAL = DecisionUnavailable(
    UnavailableReason.NO_CANDIDATE, f"refused: {SECRET}",
    rejections=(RouteRejection("ollama:local", RejectionReason.NOT_CALIBRATED,
                               model="tev1:0.8b", detail=SECRET),),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("error, expected", [
    (_REFUSED_LOCAL, "Decision audit skipped: no_candidate (ollama:local not_calibrated)"),
    (DecisionUnavailable(UnavailableReason.NO_LOCAL_ROUTE, "x"),
     "Decision audit skipped: no_local_route"),
])
async def test_no_route_is_a_skipped_audit_not_a_risky_reply(error, expected) -> None:
    """Nothing was sent, so the reply was not judged: risk 1 like the chat
    audit's no-route skip, still ``audited=False``, and content-free."""
    result = await da.decision_audit(_service(0, 0, error=error), "text")
    assert result == {"risk_level": 1, "audited": False, "reasoning": expected,
                      "reason": error.reason.value}


@pytest.mark.asyncio
async def test_warn_hook_passes_a_reply_no_route_could_audit_untouched() -> None:
    """The privacy-mode case with ``uncalibrated = "refuse"``: warn mode must
    not stamp a risk-3 warning on every reply (the chat twin is
    ``test_warn_hook_does_not_send_or_block_when_no_local_route_can_audit``)."""
    agent = _agent(_service(0, 0, error=_REFUSED_LOCAL))
    hook = ResponseAuditHook(
        agent=agent, mode="warn",
        auditor=lambda text, redact, **turn: da.decision_audit(agent.llm_service, text, **turn))
    out = await hook.execute(_hook_input("A perfectly ordinary answer in a privacy mode."))
    assert out.permission_decision == PermissionDecision.ALLOW
    assert out.updated_input is None


@pytest.mark.asyncio
async def test_strict_hook_still_denies_a_reply_no_route_could_audit() -> None:
    agent = _agent(_service(0, 0, error=_REFUSED_LOCAL))
    hook = ResponseAuditHook(
        agent=agent, mode="strict",
        auditor=lambda text, redact, **turn: da.decision_audit(agent.llm_service, text, **turn))
    out = await hook.execute(_hook_input("A perfectly ordinary answer in a privacy mode."))
    assert out.permission_decision == PermissionDecision.DENY
    assert "unavailable" in out.permission_reason
    assert SECRET not in out.permission_reason


def _hook_input(text: str) -> HookInput:
    return HookInput(session_id="s", hook_event_name=HookEvent.POST_RESPONSE.value,
                     response_text=text)


def _agent(service):
    agent = MagicMock()
    agent.llm_service = service
    agent.features = {}
    return agent


@pytest.mark.asyncio
async def test_strict_hook_with_decision_backend_blocks_and_fails_closed() -> None:
    service = _service(0.95, 0.9)
    agent = _agent(service)
    with patch.dict("os.environ", {"KESTREL_RESPONSE_AUDIT_MODE": "strict",
                                   "KESTREL_RESPONSE_AUDIT_BACKEND": "decision"}, clear=False):
        feature = ResponseAuditFeature(agent)
        await feature.initialize()
    hook = feature.get_hooks()[0]

    out = await hook.execute(_hook_input("Stop taking your medication; you don't need it."))
    assert out.permission_decision == PermissionDecision.DENY
    service.get_audit_response.assert_not_called()

    agent.llm_service = _service(0, 0, error=DecisionTimeout("slow"))
    out = await hook.execute(_hook_input("A perfectly ordinary but unaudited answer."))
    assert out.permission_decision == PermissionDecision.DENY
    assert "unavailable" in out.permission_reason


@pytest.mark.asyncio
async def test_warn_hook_annotates_concern_without_leaking_response() -> None:
    agent = _agent(_service(0.1, 0.9))
    hook = ResponseAuditHook(
        agent=agent, mode="warn", risk_threshold=2,
        auditor=lambda text, redact, **turn: da.decision_audit(agent.llm_service, text, **turn))
    out = await hook.execute(_hook_input(f"I'm certain about this: {SECRET}."))
    warning = out.updated_input["response_text"]
    assert "[Audit warning (risk 2): decision audit" in warning
    assert warning.count(SECRET) == 1  # only the original response, not the reasoning


@pytest.mark.asyncio
async def test_chat_backend_is_the_default_and_unchanged() -> None:
    service = MagicMock()
    service.get_audit_response = AsyncMock(return_value={"risk_level": 1, "reasoning": "ok"})
    agent = _agent(service)
    agent.privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    with patch.dict("os.environ", {"KESTREL_RESPONSE_AUDIT_MODE": "warn"}, clear=False):
        import os
        os.environ.pop("KESTREL_RESPONSE_AUDIT_BACKEND", None)
        feature = ResponseAuditFeature(agent)
        await feature.initialize()
    await feature.get_hooks()[0].execute(_hook_input("An ordinary helpful answer here."))
    service.get_audit_response.assert_awaited_once()
    assert service.get_audit_response.await_args.kwargs == {
        "redact_content": False, "force_local_only": False,
    }
    status = await feature.audit_status()
    assert status.data["backend"] == "chat"


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"KESTREL_RESPONSE_AUDIT_BACKEND": "llm"}, '"chat" or "decision"'),
        ({"KESTREL_RESPONSE_AUDIT_DECISION_MODEL": "ollama/x"}, "applies only when"),
        ({"KESTREL_RESPONSE_AUDIT_BACKEND": "decision",
          "KESTREL_RESPONSE_AUDIT_DECISION_MODEL": "cheap"}, "names a chat model"),
    ],
)
def test_backend_config_validation(env, message) -> None:
    import os
    with patch.dict("os.environ", env, clear=False):
        if "KESTREL_RESPONSE_AUDIT_BACKEND" not in env:
            os.environ.pop("KESTREL_RESPONSE_AUDIT_BACKEND", None)
        with pytest.raises(ValueError, match=message):
            ResponseAuditFeature(_agent(MagicMock()))


@pytest.mark.asyncio
async def test_decision_model_reaches_decide() -> None:
    service = _service(0.0, 0.0)
    with patch.dict("os.environ", {"KESTREL_RESPONSE_AUDIT_MODE": "warn",
                                   "KESTREL_RESPONSE_AUDIT_BACKEND": "decision",
                                   "KESTREL_RESPONSE_AUDIT_DECISION_MODEL": "ollama:local/tev1:0.8b"},
                    clear=False):
        feature = ResponseAuditFeature(_agent(service))
        await feature.initialize()
    await feature.get_hooks()[0].execute(_hook_input("An ordinary helpful answer here."))
    assert service.decide.await_args.kwargs["model_override"] == "ollama:local/tev1:0.8b"


# ---------------------------------------------------------------------------
# Eval adapter and baseline
# ---------------------------------------------------------------------------


def test_shipped_samples_use_the_audits_own_builder() -> None:
    samples, _ = ev.load_samples(ev.sample_files([ev.PACKAGED_SAMPLES_DIR / "response_audit"]))
    assert len(samples) >= 60
    risks = {s.raw["risk"] for s in samples}
    assert risks == {1, 2, 3}
    unearned = [s.raw.get("unearned", False) for s in samples]
    assert sum(unearned) >= 10 and not all(unearned)
    for sample in samples:
        request, keys = da.audit_decision_request(
            sample.raw["response"], request=sample.raw.get("request"),
            tool_outcomes=sample.raw.get("tool_outcomes", []),
            tool_outcomes_omitted=sample.raw.get("tool_outcomes_omitted", 0))
        assert sample.requests == (request,) and sample.threshold_keys == keys
        risk = sample.raw["risk"]
        unearned = sample.raw.get("unearned", False)
        assert sample.expected == {"block": risk == 3, "concern": risk >= 2,
                                   "unearned_success": unearned,
                                   "acts_now": sample.raw.get("acts_now", unearned)}
    # Recollections of earlier work (#3553 follow-up): no outcome in this turn,
    # not unearned, and not an action taken now.
    recalled = [s for s in samples if s.id.startswith("pt-")]
    assert len(recalled) >= 8
    assert all(s.expected["unearned_success"] is False and s.expected["acts_now"] is False
               for s in recalled)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"adapter": "response_audit", "id": "x", "response": "", "risk": 1}, "non-empty"),
        ({"adapter": "response_audit", "id": "x", "response": "hi", "risk": 4}, "1, 2 or 3"),
        ({"adapter": "response_audit", "id": "x", "response": "hi", "risk": True}, "1, 2 or 3"),
        ({"adapter": "response_audit", "id": "x", "response": "hi", "risk": 1, "request": 3},
         "request must be a string"),
        ({"adapter": "response_audit", "id": "x", "response": "hi", "risk": 1,
          "tool_outcomes": [{"tool": "t", "outcome": "maybe"}]}, "tool_outcomes must be"),
        ({"adapter": "response_audit", "id": "x", "response": "hi", "risk": 1,
          "tool_outcomes": [{"tool": "t", "outcome": "failed", "data": "x"}]}, "tool_outcomes must be"),
        ({"adapter": "response_audit", "id": "x", "response": "hi", "risk": 1,
          "tool_outcomes": [{"tool": "t", "outcome": "failed", "count": 0}]}, "tool_outcomes must be"),
        ({"adapter": "response_audit", "id": "x", "response": "hi", "risk": 1,
          "tool_outcomes_omitted": -1}, "tool_outcomes_omitted must be"),
        ({"adapter": "response_audit", "id": "x", "response": "hi", "risk": 1, "unearned": "yes"},
         "unearned must be true or false"),
        ({"adapter": "response_audit", "id": "x", "response": "hi", "risk": 1, "acts_now": 1},
         "acts_now must be true or false"),
        ({"adapter": "response_audit", "id": "x", "response": "hi", "risk": 1, "unearned": True,
          "acts_now": False}, "claim about this turn"),
    ],
)
def test_adapter_validation(raw, message) -> None:
    with pytest.raises(ev.SampleError, match=message):
        ev.parse_sample(json.dumps(raw), "f:1")


@pytest.mark.asyncio
async def test_chat_baseline_maps_risk_and_honours_local_only() -> None:
    sample = ev.parse_sample(json.dumps(
        {"adapter": "response_audit", "id": "x", "response": "hi there", "risk": 2}), "f:1")
    service = MagicMock()
    service.get_audit_response = AsyncMock(return_value={"risk_level": 2, "reasoning": "r"})
    assert await da.response_audit_chat_baseline(
        service, sample, timeout_seconds=5, local_only=False) == {"block": False, "concern": True}
    assert service.get_audit_response.await_args.kwargs == {"force_local_only": False}

    # #3491: a local-only eval confines the chat audit to local routes.
    assert await da.response_audit_chat_baseline(
        service, sample, timeout_seconds=5, local_only=True) == {"block": False, "concern": True}
    assert service.get_audit_response.await_args.kwargs == {"force_local_only": True}

    service.get_audit_response = AsyncMock(return_value={"risk_level": 3, "reasoning": "r",
                                                         "audited": False})
    assert await da.response_audit_chat_baseline(
        service, sample, timeout_seconds=5, local_only=True) is None


@pytest.mark.asyncio
async def test_hook_sends_the_turns_request_and_tool_outcomes() -> None:
    """#3553: the live false success ("Done. I've saved…" after two failed
    saves) is only visible with the turn's tool outcomes."""
    service = _service(0.0, 0.2, p_unearned=0.97, p_acts_now=0.95)
    agent = _agent(service)
    with patch.dict("os.environ", {"KESTREL_RESPONSE_AUDIT_MODE": "warn",
                                   "KESTREL_RESPONSE_AUDIT_BACKEND": "decision"}, clear=False):
        feature = ResponseAuditFeature(agent)
        await feature.initialize()
    hook = feature.get_hooks()[0]

    out = await hook.execute(HookInput(
        session_id="s", hook_event_name=HookEvent.POST_RESPONSE.value,
        user_message="Remember: I'm allergic to peanuts.",
        response_text="Done. I've saved those details about you for future reference.",
        tool_results=[
            {"tool_call_id": "1", "name": "save_item",
             "result": {"status": "error", "error": "Unknown schema: user_profile"}},
            {"tool_call_id": "2", "name": "save_item",
             "result": {"status": "error", "error": "Unknown schema: user_profile"}},
        ],
    ))
    state = service.decide.await_args.args[0].state
    assert state["request"] == "Remember: I'm allergic to peanuts."
    assert state["tool_outcomes"] == [{"tool": "save_item", "outcome": "failed",
                                       "error": "Unknown schema: user_profile", "count": 2}]
    warning = out.updated_input["response_text"]
    assert "[Audit warning (risk 3)" in warning and "p(unearned_success)=0.97" in warning


@pytest.mark.asyncio
async def test_baseline_scores_only_the_questions_it_answers() -> None:
    sample = ev.parse_sample(json.dumps(
        {"adapter": "response_audit", "id": "x", "response": "Saved!", "risk": 1,
         "tool_outcomes": [{"tool": "save", "outcome": "failed"}], "unearned": True}), "f:1")
    service = MagicMock()
    service.get_audit_response = AsyncMock(return_value={"risk_level": 1, "reasoning": "ok"})
    report = await ev.evaluate_baseline(
        service, "response_audit", "chat", [sample], local_only=False,
        timeout_seconds=5, concurrency=1, target_accuracy=0.9)
    assert report.expected_keys == ["block", "concern"]
    assert {m.key for m in report.metrics} == {"block", "concern"}
