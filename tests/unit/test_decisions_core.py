"""Decisions modality (#3424): config, override grammar, normaliser, thresholds,
fit, and route resolution. Pure functions; no service, no network."""

from __future__ import annotations

import pytest

from kestrel_sdk.llm.decisions import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionModelInfo,
    DecisionProtocolError,
    DecisionRequest,
    DecisionRequestInvalid,
    DecisionUnavailable,
    NoulQuestion,
    RejectionReason,
    ScoreQuestion,
    UnavailableReason,
    validate_decision_request,
)

from kestrel_sovereign.llm.decisions.config import (
    DecisionRouteConfig,
    parse_decision_selector,
    parse_route_decision_config,
    parse_service_decision_config,
)
from kestrel_sovereign.llm.decisions.fit import FRAMING_TOKENS, RequestFootprint, misfit
from kestrel_sovereign.llm.decisions.normalize import normalize_response
from kestrel_sovereign.llm.decisions.resolve import (
    PinStatus,
    RouteDecisionState,
    candidate_routes,
    select_candidate,
)
from kestrel_sovereign.llm.decisions.thresholds import ThresholdBook


def _snapshot(**questions):
    return validate_decision_request(
        DecisionRequest(
            state={"ticket": "charged twice"},
            questions=questions
            or {
                "team": ChoiceQuestion(
                    instructions="Which team?", options={"billing": None, "tech": None}
                )
            },
        )
    )


# ---------------------------------------------------------------------------
# Config and grammar
# ---------------------------------------------------------------------------


def test_route_config_defaults_and_validation() -> None:
    assert parse_route_decision_config("ollama", "local", {}) == DecisionRouteConfig()
    pinned = parse_route_decision_config(
        "ollama", "local", {"decision_model": "nimble", "decision_context_limit": 8192}
    )
    assert pinned == DecisionRouteConfig(pin="nimble", context_limit=8192)
    with pytest.raises(ValueError, match="only to a pinned"):
        parse_route_decision_config("ollama", "local", {"decision_context_limit": 8192})
    with pytest.raises(ValueError, match="decision_hints"):
        parse_route_decision_config("ollama", "local", {"decision_hints": "nimble"})
    with pytest.raises(ValueError, match="decision_model"):
        parse_route_decision_config("ollama", "local", {"decision_model": ""})


def test_service_config() -> None:
    assert parse_service_decision_config({}).route is None
    assert parse_service_decision_config({"decision_route": "none"}).disabled
    assert parse_service_decision_config({"decision_route": "ollama:local"}).route == "ollama:local"
    with pytest.raises(ValueError, match="names a route"):
        parse_service_decision_config({"decision_route": "ollama/nimble"})
    with pytest.raises(ValueError, match="canary"):
        parse_service_decision_config({"decision_canary_timeout_seconds": 0})


@pytest.mark.parametrize(
    ("raw", "vendor", "route", "model"),
    [
        ("ollama", "ollama", None, None),
        ("ollama:local", "ollama", "local", None),
        ("openrouter/typesafe/jev-1.13", "openrouter", None, "typesafe/jev-1.13"),
        ("ollama:local/nimble:9b", "ollama", "local", "nimble:9b"),
    ],
)
def test_selector_grammar(raw, vendor, route, model) -> None:
    selector = parse_decision_selector(raw)
    assert (selector.vendor, selector.route, selector.model) == (vendor, route, model)


@pytest.mark.parametrize("raw", ["cheap", "", "ollama:", ":local", "a:b:c/m", "ollama/"])
def test_selector_grammar_rejects(raw) -> None:
    with pytest.raises(ValueError):
        parse_decision_selector(raw)


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------


def test_normaliser_recomputes_argmax_and_score_and_drops_confidence() -> None:
    snapshot = _snapshot(
        team=ChoiceQuestion(instructions="Which?", options={"billing": None, "tech": None}),
        urgency=ScoreQuestion(instructions="How urgent?", levels=["low", "mid", "high"]),
        refund=NoulQuestion(instructions="Refund asked?"),
    )
    body = {
        "answers": {
            "team": {"type": "choice", "choice": "tech", "confidence": 0.99,
                     "probabilities": {"billing": 0.7, "tech": 0.3}},
            "urgency": {"type": "score", "score": 99, "legend": {},
                        "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}},
            "refund": {"type": "noul", "noul": 0.9},
        },
        "usage": {"input_tokens": 120, "cost": 0.0001},
    }
    result = normalize_response(snapshot, body)

    team = result.answers["team"]
    assert isinstance(team, ChoiceAnswer) and team.choice == "billing"
    assert not hasattr(team, "confidence")
    assert result.answers["urgency"].score == pytest.approx(0.3 + 2 * 0.5)
    assert result.answers["refund"].p_true == pytest.approx(0.9)
    assert (result.input_tokens, result.cost_usd, result.usage_available) == (120, 0.0001, True)


def test_normaliser_renormalises_small_drift() -> None:
    snapshot = _snapshot()
    body = {"answers": {"team": {"probabilities": {"billing": 0.6004, "tech": 0.4}}}}
    probs = normalize_response(snapshot, body).answers["team"].probabilities
    assert sum(probs.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "answers",
    [
        {},
        {"team": {"probabilities": {"billing": 1.0}}},
        {"team": {"probabilities": {"billing": 0.5, "tech": 0.5, "x": 0.0}}},
        {"team": {"probabilities": {"billing": 0.9, "tech": 0.3}}},
        {"team": {"probabilities": {"billing": float("nan"), "tech": 0.5}}},
        {"team": {"probabilities": {"billing": "0.5", "tech": 0.5}}},
        {"team": {"type": "noul", "noul": 0.5}},
        {"team": {"probabilities": {"billing": 0.5, "tech": 0.5}}, "extra": {}},
    ],
)
def test_normaliser_refuses_the_whole_result(answers) -> None:
    with pytest.raises(DecisionProtocolError):
        normalize_response(_snapshot(), {"answers": answers})


def test_normaliser_without_usage_marks_usage_unavailable() -> None:
    body = {"answers": {"team": {"probabilities": {"billing": 0.5, "tech": 0.5}}}}
    result = normalize_response(_snapshot(), body)
    assert (result.input_tokens, result.usage_available) == (None, False)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


BOOK = ThresholdBook.from_config(
    {
        "thresholds": {
            "gate": {
                "uncalibrated": "refuse",
                "models": {"ollama:local/nimble": {"a": 0.6, "b": 0.7}},
            },
            "soft": {
                "uncalibrated": "default",
                "default": {"a": 0.5, "b": 0.5},
                "models": {"ollama:local/nimble": {"a": 0.62}},
            },
        }
    }
)


def test_calibration_is_all_or_nothing() -> None:
    assert BOOK.admits("gate", "ollama:local/nimble", ["a", "b"])
    assert not BOOK.admits("gate", "ollama:local/nimble", ["a", "b", "c"])
    assert not BOOK.admits("gate", "openrouter:api/jev", ["a"])
    # "soft" has a partial entry for nimble: the request asking both falls
    # back to defaults for BOTH questions, never a mix.
    resolution = BOOK.resolve("soft", "ollama:local/nimble", ["a", "b"])
    assert resolution.calibrated is False
    assert dict(resolution.thresholds) == {"a": 0.5, "b": 0.5}
    assert BOOK.resolve("soft", "ollama:local/nimble", ["a"]).thresholds["a"] == 0.62


def test_callers_without_tables_are_not_applicable() -> None:
    resolution = BOOK.resolve("nobody", "x/y", ["a"])
    assert resolution.calibrated is None and dict(resolution.thresholds) == {}
    assert BOOK.admits("nobody", "x/y", ["a"])


def test_default_table_must_cover_the_request() -> None:
    BOOK.check_request("soft", ["a", "b"])
    with pytest.raises(DecisionRequestInvalid, match="no threshold"):
        BOOK.check_request("soft", ["a", "z"])


@pytest.mark.parametrize(
    "table",
    [
        {"uncalibrated": "maybe"},
        {"default": {"a": 1.5}},
        {"default": {"a": True}},
        {"models": {"x/y": "0.5"}},
    ],
)
def test_threshold_config_validation(table) -> None:
    with pytest.raises(ValueError):
        ThresholdBook.from_config({"thresholds": {"c": table}})


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def _info(**kw) -> DecisionModelInfo:
    base = dict(id="m", vendor="v", route="v:r", context_limit=8192)
    base.update(kw)
    return DecisionModelInfo(**base)


def test_fit_uses_per_question_size_and_route_caps() -> None:
    small = RequestFootprint.of(_snapshot())
    assert misfit(_info(), small) is None
    assert misfit(_info(context_limit=None), small) == "context limit unknown"
    assert misfit(_info(context_limit=None), small, context_limit_override=8192) is None
    assert "tokens per question" in misfit(_info(context_limit=FRAMING_TOKENS), small)

    big_state = validate_decision_request(
        DecisionRequest(
            state="x" * 40_000,
            questions={"q": NoulQuestion(instructions="ok?")},
        )
    )
    big = RequestFootprint.of(big_state)
    assert "tokens per question" in misfit(_info(context_limit=8192), big)
    assert misfit(_info(context_limit=32000), big) is None
    assert "bytes" in misfit(_info(context_limit=32000, max_request_bytes=16_000), big)


def test_fit_question_and_option_caps() -> None:
    many_options = _snapshot(
        q=ChoiceQuestion(instructions="pick", options={f"o{i}": None for i in range(5)})
    )
    footprint = RequestFootprint.of(many_options)
    assert "options" in misfit(_info(max_options=4), footprint)
    assert "questions" in misfit(_info(max_questions=0), footprint)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _route(name, *, local=False, decisions=True, models=None, pin=None,
           pin_status=PinStatus.UNCHECKED, hints=()):
    vendor, _, route = name.partition(":")
    state = RouteDecisionState(config=DecisionRouteConfig(pin=pin, hints=tuple(hints)))
    if models is not None:
        state.record_discovery([_info(id=m, vendor=vendor, route=name) for m in models])
    if pin is not None:
        state.pin_status = pin_status
        if pin_status is PinStatus.VERIFIED:
            state.pin_info = _info(id=pin, vendor=vendor, route=name)
    return {
        "name": name, "vendor": vendor, "route": route, "is_local": local,
        "capabilities": {"supports_decisions": decisions}, "decision_state": state,
    }


SVC = parse_service_decision_config({})


def _select(routes, selector=None, thresholds=BOOK, caller="anyone", snapshot=None):
    snapshot = snapshot or _snapshot()
    return select_candidate(
        routes,
        footprint=RequestFootprint.of(snapshot),
        selector=selector,
        thresholds=thresholds,
        caller=caller,
        question_ids=tuple(snapshot.questions),
    )


def test_privacy_filter_runs_before_anything_else() -> None:
    cloud, local = _route("openrouter:api"), _route("ollama:local", local=True)
    assert candidate_routes([cloud, local], service_config=SVC, selector=None, local_only=True) == [local]
    with pytest.raises(DecisionUnavailable) as excinfo:
        candidate_routes([cloud], service_config=SVC, selector=None, local_only=True)
    assert excinfo.value.reason is UnavailableReason.NO_LOCAL_ROUTE


def test_routes_without_decision_support_are_not_candidates() -> None:
    chat_only = _route("anthropic:api", decisions=False)
    with pytest.raises(DecisionUnavailable) as excinfo:
        candidate_routes(
            [chat_only],
            service_config=parse_service_decision_config({"decision_route": "anthropic"}),
            selector=None, local_only=False,
        )
    assert excinfo.value.reason is UnavailableReason.NO_ROUTE


def test_disabled_and_selector_conflict() -> None:
    routes = [_route("ollama:local", local=True), _route("openrouter:api")]
    with pytest.raises(DecisionUnavailable) as excinfo:
        candidate_routes(routes, service_config=parse_service_decision_config({"decision_route": "none"}),
                         selector=None, local_only=False)
    assert excinfo.value.reason is UnavailableReason.DISABLED
    with pytest.raises(DecisionUnavailable) as excinfo:
        candidate_routes(routes, service_config=parse_service_decision_config({"decision_route": "ollama:local"}),
                         selector=parse_decision_selector("openrouter/x"), local_only=False)
    assert excinfo.value.reason is UnavailableReason.SELECTOR_CONFLICT


def test_bare_model_override_matches_no_route_with_a_clear_error() -> None:
    with pytest.raises(DecisionUnavailable, match="bare model ids") as excinfo:
        candidate_routes([_route("ollama:local", local=True)], service_config=SVC,
                         selector=parse_decision_selector("nimble"), local_only=False)
    assert excinfo.value.reason is UnavailableReason.NO_ROUTE


def test_one_ambiguous_route_never_blocks_a_usable_one() -> None:
    ambiguous = _route("openrouter:api", models=["a", "b"])
    usable = _route("ollama:local", local=True, models=["nimble"])
    candidate = _select([ambiguous, usable])
    assert candidate.provider is usable and candidate.info.id == "nimble"

    with pytest.raises(DecisionUnavailable) as excinfo:
        _select([ambiguous, _route("x:y", models=[])])
    assert excinfo.value.reason is UnavailableReason.NO_CANDIDATE
    reasons = [r.reason for r in excinfo.value.rejections]
    assert reasons == [RejectionReason.AMBIGUOUS_MODEL, RejectionReason.NO_MODELS]


def test_hints_narrow_and_pins_are_authoritative() -> None:
    hinted = _route("openrouter:api", models=["typesafe/jev-1.13", "liquid/d1"], hints=["jev"])
    assert _select([hinted]).info.id == "typesafe/jev-1.13"

    unverified = _route("ollama:local", local=True, models=["nimble"], pin="nimble",
                        pin_status=PinStatus.UNVERIFIED)
    with pytest.raises(DecisionUnavailable) as excinfo:
        _select([unverified])
    assert excinfo.value.rejections[0].reason is RejectionReason.UNVERIFIED_PIN


def test_override_against_pins_and_discovery() -> None:
    pinned = _route("ollama:local", local=True, pin="nimble", pin_status=PinStatus.VERIFIED)
    with pytest.raises(DecisionUnavailable) as excinfo:
        _select([pinned], selector=parse_decision_selector("ollama:local/tev1"))
    assert excinfo.value.rejections[0].reason is RejectionReason.PIN_CONFLICT
    assert _select([pinned], selector=parse_decision_selector("ollama:local/nimble")).info.id == "nimble"

    served = _route("openrouter:api", models=["a", "b"])
    assert _select([served], selector=parse_decision_selector("openrouter/b")).info.id == "b"
    with pytest.raises(DecisionUnavailable) as excinfo:
        _select([served], selector=parse_decision_selector("openrouter/c"))
    assert excinfo.value.rejections[0].reason is RejectionReason.NOT_SERVED
    with pytest.raises(DecisionUnavailable) as excinfo:
        _select([_route("openrouter:api", models=[])], selector=parse_decision_selector("openrouter/c"))
    assert excinfo.value.rejections[0].reason is RejectionReason.NO_MODELS


def test_refuse_policy_rejects_uncalibrated_models_before_dispatch() -> None:
    snapshot = validate_decision_request(
        DecisionRequest(state="s", questions={"a": NoulQuestion(instructions="a?"),
                                              "b": NoulQuestion(instructions="b?")})
    )
    cloud = _route("openrouter:api", models=["jev"])
    local = _route("ollama:local", local=True, models=["nimble"])
    candidate = _select([cloud, local], caller="gate", snapshot=snapshot)
    assert candidate.provider is local
    with pytest.raises(DecisionUnavailable) as excinfo:
        _select([cloud], caller="gate", snapshot=snapshot)
    assert excinfo.value.rejections[0].reason is RejectionReason.NOT_CALIBRATED


def test_no_fit_is_recorded_per_route() -> None:
    tiny = _route("ollama:local", local=True, models=["tev1"])
    tiny["decision_state"].models = (_info(id="tev1", route="ollama:local", context_limit=100),)
    with pytest.raises(DecisionUnavailable) as excinfo:
        _select([tiny])
    assert excinfo.value.rejections[0].reason is RejectionReason.NO_FIT


# ---------------------------------------------------------------------------
# Registry → route dict
# ---------------------------------------------------------------------------


def test_route_decision_config_crosses_the_sdk_boundary(monkeypatch) -> None:
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry
    from kestrel_sovereign.llm.service import LLMService

    registry = ProviderRegistry({})
    info = registry._build_route(
        "ollama", "local", {"is_cloud": False},
        {"adapter": "OllamaAdapter", "host": "http://ollama.test:11434",
         "decision_model": "nimble:9b", "decision_hints": ["nimble"],
         "decision_context_limit": 8192},
    )
    service = LLMService.__new__(LLMService)
    [route] = service._convert_providers_format([info])
    state = route["decision_state"]
    assert isinstance(state, RouteDecisionState)
    assert state.config == DecisionRouteConfig(pin="nimble:9b", hints=("nimble",), context_limit=8192)
    assert state.needs_discovery
    assert route["capabilities"]["supports_decisions"] is True


def test_invalid_route_decision_config_fails_at_build() -> None:
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    with pytest.raises(ValueError, match="decision_context_limit"):
        ProviderRegistry({})._build_route(
            "ollama", "local", {"is_cloud": False},
            {"adapter": "OllamaAdapter", "decision_context_limit": 8192},
        )
