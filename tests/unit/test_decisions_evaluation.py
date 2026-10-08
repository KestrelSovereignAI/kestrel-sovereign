"""Decisions eval harness (#3434, spec §9): samples, metrics, threshold
proposals, the per-model runner, and the CLI's candidate and privacy rules."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List

import pytest

from kestrel_sdk.llm.decisions import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionResult,
    DecisionTransportError,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

from kestrel_sovereign import cli_decisions
from kestrel_sovereign.llm.decisions import evaluation as ev


def _line(**overrides: Any) -> str:
    sample = {
        "id": "s1",
        "state": {"q": "pet name?", "c": "Quasar the axolotl"},
        "questions": {"c0": {"type": "noul", "instructions": "Does `c` answer `q`?"}},
        "expected": {"c0": True},
        "threshold_keys": {"c0": "answers"},
    }
    sample.update(overrides)
    return json.dumps(sample)


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def test_parse_sample_builds_sdk_questions() -> None:
    sample = ev.parse_sample(json.dumps({
        "id": "x",
        "state": "s",
        "questions": {
            "n": {"type": "noul", "instructions": "n?", "criteria": {"true": "yes"}},
            "c": {"type": "choice", "instructions": "c?", "criteria": {"a": None, "b": "B"}},
            "s": {"type": "score", "instructions": "s?", "criteria": ["lo", "hi"]},
        },
        "expected": {"n": False, "c": "b", "s": 1},
    }), "f:1")
    assert isinstance(sample.questions["n"], NoulQuestion)
    assert sample.questions["n"].true_means == "yes"
    assert dict(sample.questions["c"].options) == {"a": None, "b": "B"}
    assert list(sample.questions["s"].levels) == ["lo", "hi"]
    assert sample.threshold_keys == {}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": ""}, "non-empty string id"),
        ({"questions": {}}, "non-empty object"),
        ({"expected": {}}, "exactly the questions"),
        ({"expected": {"c0": "yes"}}, "true/false"),
        ({"questions": {"c0": {"type": "choice", "instructions": "x", "criteria": {"a": None}}},
          "expected": {"c0": "z"}}, "not an option"),
        ({"questions": {"c0": {"type": "score", "instructions": "x", "criteria": ["a", "b"]}},
          "expected": {"c0": 2}}, "level index"),
        ({"questions": {"c0": {"type": "rank", "instructions": "x"}}}, "unknown question type"),
    ],
)
def test_parse_sample_rejects_malformed(overrides, message) -> None:
    with pytest.raises(ev.SampleError, match=message):
        ev.parse_sample(_line(**overrides), "f:1")


def test_set_hash_follows_what_is_sent_not_file_bytes(tmp_path: Path, monkeypatch) -> None:
    a = tmp_path / "a.jsonl"
    a.write_text(_line() + "\n")
    _, digest = ev.load_samples([a])
    # Reformatting the same sample is the same evidence.
    a.write_text(json.dumps(json.loads(_line()), indent=2).replace("\n", " ") + "\n")
    assert ev.load_samples([a])[1] == digest
    # A label edit is not.
    a.write_text(_line(expected={"c0": False}) + "\n")
    assert ev.load_samples([a])[1] != digest

    # A caller's prompt change is new evidence even though its file is unchanged.
    from kestrel_sovereign.storage import memory_answerability as ma

    shipped = ev.sample_files([ev.PACKAGED_SAMPLES_DIR / "memory_answerability"])
    _, before = ev.load_samples(shipped)
    monkeypatch.setattr(ma, "_DECISION_INSTRUCTIONS", ma._DECISION_INSTRUCTIONS + " Be strict.")
    assert ev.load_samples(shipped)[1] != before


def test_load_samples_hashes_content_and_refuses_duplicates(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    a.write_text(_line(id="s1") + "\n\n" + _line(id="s2") + "\n")
    samples, digest = ev.load_samples(ev.sample_files([tmp_path]))
    assert [s.id for s in samples] == ["s1", "s2"]
    assert ev.load_samples([a])[1] == digest

    a.write_text(_line(id="s1") + "\n" + _line(id="s3") + "\n")
    assert ev.load_samples([a])[1] != digest

    a.write_text(_line(id="s1") + "\n" + _line(id="s1") + "\n")
    with pytest.raises(ev.SampleError, match="duplicate"):
        ev.load_samples([a])
    with pytest.raises(ev.SampleError, match="no such"):
        ev.sample_files([tmp_path / "missing"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _noul(p: float, expected: bool, key: str = "answers") -> ev.Observation:
    return ev.Observation(key, NoulQuestion(instructions="?"), expected, NoulAnswer(p_true=p))


def test_noul_threshold_maximises_youden_and_reports_precision_recall() -> None:
    observations = [_noul(p, True) for p in (0.9, 0.8, 0.35)] + [
        _noul(p, False) for p in (0.3, 0.2, 0.1)
    ]
    [m] = ev.score_observations(observations)
    assert m.kind == "noul" and m.n == 6
    assert m.proposed_threshold == 0.35  # separates perfectly; 0.5 would miss 0.35
    assert m.accuracy == 1.0 and m.extra["precision"] == 1.0 and m.extra["recall"] == 1.0
    assert m.extra["accuracy@0.5"] == pytest.approx(5 / 6)
    expected_brier = sum((p - y) ** 2 for p, y in
                         [(0.9, 1), (0.8, 1), (0.35, 1), (0.3, 0), (0.2, 0), (0.1, 0)]) / 6
    assert m.brier == pytest.approx(expected_brier)
    assert 0.0 <= m.ece <= 1.0


def test_single_class_noul_cannot_propose_a_threshold() -> None:
    [m] = ev.score_observations([_noul(0.9, True), _noul(0.7, True)])
    assert m.proposed_threshold is None and "both true and false" in m.note


def test_choice_threshold_meets_target_accuracy_with_coverage() -> None:
    question = ChoiceQuestion(instructions="?", options={"a": None, "b": None})

    def obs(pa: float, truth: str) -> ev.Observation:
        choice = "a" if pa >= 0.5 else "b"
        return ev.Observation("team", question, truth,
                              ChoiceAnswer(choice=choice, probabilities={"a": pa, "b": 1 - pa}))

    observations = [obs(0.95, "a"), obs(0.9, "a"), obs(0.6, "b"), obs(0.55, "a")]
    [m] = ev.score_observations(observations, target_accuracy=0.9)
    assert m.kind == "choice" and m.accuracy == 0.75
    assert m.proposed_threshold == 0.9 and m.extra["coverage"] == 0.5

    [m] = ev.score_observations([obs(0.6, "b")], target_accuracy=0.9)
    assert m.proposed_threshold is None and "no cut" in m.note


def test_score_metrics_include_mae() -> None:
    question = ScoreQuestion(instructions="?", levels=["lo", "mid", "hi"])
    observations = [
        ev.Observation("urgency", question, 2, ScoreAnswer(score=1.8, probabilities=(0.0, 0.2, 0.8))),
        ev.Observation("urgency", question, 0, ScoreAnswer(score=0.4, probabilities=(0.7, 0.2, 0.1))),
    ]
    [m] = ev.score_observations(observations)
    assert m.kind == "score" and m.accuracy == 1.0
    assert m.extra["mae"] == pytest.approx((0.2 + 0.4) / 2)


def test_a_threshold_key_cannot_mix_question_types() -> None:
    with pytest.raises(ev.SampleError, match="mixes question types"):
        ev.score_observations([
            _noul(0.9, True, key="k"),
            ev.Observation("k", ScoreQuestion(instructions="?", levels=["a", "b"]), 0,
                           ScoreAnswer(score=0.0, probabilities=(1.0, 0.0))),
        ])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class _FakeService:
    def __init__(self, p_by_id: Dict[str, float], fail_ids=()):
        self.p_by_id = p_by_id
        self.fail_ids = set(fail_ids)
        self.calls: List[Dict[str, Any]] = []

    async def decide(self, request, **kwargs):
        self.calls.append(kwargs)
        sample_id = request.state["id"]
        if sample_id in self.fail_ids:
            raise DecisionTransportError("down")
        return DecisionResult(
            answers=MappingProxyType({q: NoulAnswer(p_true=self.p_by_id[sample_id])
                                      for q in request.questions}),
            vendor="ollama", route="ollama:local", model="tev1",
            thresholds=MappingProxyType({}), calibrated=None, input_tokens=10,
            duration_ms=100 + len(self.calls),
        )


def _samples(labels: Dict[str, bool]):
    return [ev.parse_sample(_line(id=i, state={"id": i}, expected={"c0": y}), f"f:{n}")
            for n, (i, y) in enumerate(labels.items(), start=1)]


@pytest.mark.asyncio
async def test_evaluate_model_runs_under_the_eval_caller_and_counts_errors() -> None:
    service = _FakeService({"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1}, fail_ids={"e"})
    samples = _samples({"a": True, "b": True, "c": False, "d": False, "e": True})
    report = await ev.evaluate_model(
        service, "ollama:local/tev1", samples, caller="memory_answerability",
        local_only=True, timeout_seconds=5, concurrency=2, target_accuracy=0.9,
    )
    assert {c["caller"] for c in service.calls} == {"kestrel.eval.memory_answerability"}
    assert {c["model_override"] for c in service.calls} == {"ollama:local/tev1"}
    assert all(c["local_only"] is True for c in service.calls)
    assert all(c["threshold_keys"] == {"c0": "answers"} for c in service.calls)
    assert report.errors == {"DecisionTransportError": 1}
    assert len(report.latencies_ms) == 4
    [m] = report.metrics
    assert m.key == "answers" and m.accuracy == 1.0 and m.proposed_threshold is not None


def _multi_sample(sample_id: str, labels: Dict[str, bool]) -> ev.Sample:
    """One call site sending a one-question request per label, in parallel."""

    from kestrel_sdk.llm.decisions import DecisionRequest

    return ev.Sample(
        id=sample_id,
        requests=tuple(
            DecisionRequest(state={"id": f"{sample_id}.{qid}"},
                            questions={qid: NoulQuestion(instructions="?")})
            for qid in labels
        ),
        expected=dict(labels),
        threshold_keys={qid: "answers" for qid in labels},
        source="f:1",
    )


@pytest.mark.asyncio
async def test_a_multi_request_sample_is_scored_as_one_call_site() -> None:
    service = _FakeService({"s.c0": 0.9, "s.c1": 0.1, "t.c0": 0.9}, fail_ids={"t.c1"})
    report = await ev.evaluate_model(
        service, "ollama:local/tev1",
        [_multi_sample("s", {"c0": True, "c1": False}),
         _multi_sample("t", {"c0": True, "c1": True})],
        caller="memory_answerability", local_only=True, timeout_seconds=5,
        concurrency=1, target_accuracy=0.9,
    )
    # Each request carries only its own question's threshold key.
    assert sorted(map(str, (c["threshold_keys"] for c in service.calls))) == sorted(
        [str({"c0": "answers"}), str({"c1": "answers"})] * 2)
    # One failed request fails the whole call site, which contributes nothing.
    assert report.errors == {"DecisionTransportError": 1}
    [m] = report.metrics
    assert m.key == "answers" and m.n == 2 and m.accuracy == 1.0
    # Its latency is the slowest of its parallel requests.
    assert len(report.latencies_ms) == 1 and report.latencies_ms[0] >= 101


@pytest.mark.asyncio
async def test_a_samples_requests_run_together_under_any_concurrency() -> None:
    import asyncio

    in_flight = 0
    peak = 0

    class _Slow(_FakeService):
        async def decide(self, request, **kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return await super().decide(request, **kwargs)

    labels = {f"c{i}": True for i in range(3)}
    service = _Slow({f"s.c{i}": 0.9 for i in range(3)} | {f"t.c{i}": 0.9 for i in range(3)})
    await ev.evaluate_model(
        service, "ollama:local/tev1", [_multi_sample("s", labels), _multi_sample("t", labels)],
        caller="memory_answerability", local_only=True, timeout_seconds=5,
        concurrency=1, target_accuracy=0.9,
    )
    # One call site at a time, its three requests together: never serialized.
    assert peak == 3


def test_a_sample_needs_requests_with_distinct_question_ids() -> None:
    from kestrel_sdk.llm.decisions import DecisionRequest

    request = DecisionRequest(state={}, questions={"c0": NoulQuestion(instructions="?")})
    with pytest.raises(ev.SampleError, match="repeat across"):
        ev.Sample(id="x", requests=(request, request), expected={"c0": True},
                  threshold_keys={}, source="f:1")
    with pytest.raises(ev.SampleError, match="needs a request"):
        ev.Sample(id="x", requests=(), expected={}, threshold_keys={}, source="f:1")


def test_report_and_snippet_rendering() -> None:
    full = ev.ModelReport("ollama:local/tev1", "ollama:local", "tev1",
                          metrics=[ev.KeyMetrics("answers", "noul", 4, 1.0, 0.02, 0.05, 0.42)],
                          latencies_ms=[100, 120, 140, 400], expected_keys=["answers"])
    partial = ev.ModelReport("openrouter:api/x", "openrouter:api", "x",
                             metrics=[ev.KeyMetrics("answers", "noul", 2, 0.5, 0.3, 0.2, None,
                                                    note="needs both")])
    text = ev.render_report([full, partial], samples=4, sample_hash="ab" * 32)
    assert "ollama:local/tev1" in text and "p95=400ms" in text and "needs both" in text

    snippet = ev.render_threshold_snippet([full, partial], caller="memory_answerability",
                                          samples=4, sample_hash="ab" * 32,
                                          today=date(2026, 10, 2))
    assert '[decisions.thresholds."memory_answerability".models."ollama:local/tev1"]' in snippet
    assert '"answers" = 0.42' in snippet and "2026-10-02" in snippet and "abababababab" in snippet
    assert "openrouter:api/x" not in snippet  # partial calibration is never offered


# ---------------------------------------------------------------------------
# CLI rules
# ---------------------------------------------------------------------------


ROUTES = [
    {"route": "openrouter:api", "vendor": "openrouter", "is_local": False, "pin": None,
     "pin_status": "unchecked", "models": [{"id": "typesafe/jev-1.13"}, {"id": "liquid/d1"}]},
    {"route": "ollama:local", "vendor": "ollama", "is_local": True, "pin": "nimble",
     "pin_status": "verified", "models": [{"id": "nimble"}, {"id": "tev1"}]},
    {"route": "ollama:box", "vendor": "ollama", "is_local": True, "pin": "x",
     "pin_status": "unverified", "models": [{"id": "x"}]},
]


def test_candidate_selectors_respect_pins_filters_and_privacy() -> None:
    assert cli_decisions.candidate_selectors(ROUTES, route_filters=None, model_filters=None,
                                             local_only=False) == [
        "openrouter:api/typesafe/jev-1.13", "openrouter:api/liquid/d1", "ollama:local/nimble",
    ]
    assert cli_decisions.candidate_selectors(ROUTES, route_filters=None, model_filters=None,
                                             local_only=True) == ["ollama:local/nimble"]
    assert cli_decisions.candidate_selectors(ROUTES, route_filters=["openrouter"],
                                             model_filters=["jev"], local_only=False) == [
        "openrouter:api/typesafe/jev-1.13"]


class _CliService:
    def __init__(self):
        self.seen_local_only: List[bool] = []

    async def reconcile_decision_capabilities(self, use_cache=True):
        return None

    def describe_decision_routes(self):
        return ROUTES


@pytest.mark.asyncio
async def test_private_samples_default_to_local_routes(tmp_path: Path, monkeypatch, capsys) -> None:
    private = tmp_path / "mine.jsonl"
    private.write_text(_line() + "\n")
    seen: List[Dict[str, Any]] = []

    async def fake_eval(service, selector, samples, **kwargs):
        seen.append({"selector": selector, **kwargs})
        return ev.ModelReport(selector, *selector.split("/", 1))

    monkeypatch.setattr(cli_decisions, "evaluate_model", fake_eval)

    def args(**kw):
        base = dict(caller="memory_answerability", samples=[private], route=None, model=None,
                    local_only=False, allow_cloud=False, timeout=5.0, concurrency=1,
                    target_accuracy=0.9, json=None, baseline=None)
        base.update(kw)
        return argparse.Namespace(**base)

    assert await cli_decisions._eval(_CliService(), args()) == 0
    assert [s["selector"] for s in seen] == ["ollama:local/nimble"]
    assert all(s["local_only"] for s in seen)
    assert "local routes only" in capsys.readouterr().err

    seen.clear()
    assert await cli_decisions._eval(_CliService(), args(allow_cloud=True)) == 0
    assert "openrouter:api/typesafe/jev-1.13" in [s["selector"] for s in seen]
    assert not any(s["local_only"] for s in seen)


def test_snippet_requires_a_complete_run_and_parses_as_toml() -> None:
    import tomllib

    good = ev.ModelReport("ollama:local/tev1", "ollama:local", "tev1",
                          metrics=[ev.KeyMetrics("answer.score", "noul", 4, 1.0, 0.0, 0.0, 0.4)],
                          expected_keys=["answer.score"])
    errored = ev.ModelReport("a:b/m", "a:b", "m", errors={"DecisionTimeout": 1},
                             metrics=[ev.KeyMetrics("answers", "noul", 4, 1.0, 0.0, 0.0, 0.4)],
                             expected_keys=["answers"])
    uncovered = ev.ModelReport("c:d/m", "c:d", "m",
                               metrics=[ev.KeyMetrics("a", "noul", 4, 1.0, 0.0, 0.0, 0.4)],
                               expected_keys=["a", "b"])
    assert errored.proposal_blocker() == "some samples failed"
    assert "b" in uncovered.proposal_blocker()

    snippet = ev.render_threshold_snippet([good, errored, uncovered], caller="x.y",
                                          samples=4, sample_hash="cd" * 32)
    parsed = tomllib.loads(snippet)
    assert parsed["decisions"]["thresholds"]["x.y"]["models"]["ollama:local/tev1"] == {"answer.score": 0.4}
    assert parsed["decisions"]["thresholds"]["x.y"]["uncalibrated"] == "refuse"

    # Pasted on its own, the proposal is usable: the calibrated model admits
    # requests and nothing is refused for a missing default table.
    from kestrel_sovereign.llm.decisions.thresholds import ThresholdBook

    book = ThresholdBook.from_config(parsed["decisions"])
    book.check_request("x.y", ["answer.score"])
    assert book.admits("x.y", "ollama:local/tev1", ["answer.score"])
    assert not book.admits("x.y", "other:route/m", ["answer.score"])
    assert "a:b/m" not in snippet and "c:d/m" not in snippet
    text = ev.render_report([errored], samples=4, sample_hash="cd" * 32)
    assert "no calibration proposal: some samples failed" in text


@pytest.mark.asyncio
async def test_default_samples_must_exist(tmp_path: Path, capsys) -> None:
    args = argparse.Namespace(caller="no_such_caller", samples=None, route=None, model=None,
                              local_only=False, allow_cloud=False, timeout=5.0, concurrency=1,
                              target_accuracy=0.9, json=None, baseline=None)
    assert await cli_decisions._eval(_CliService(), args) == 2
    assert "no shipped sample set" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Caller adapters and baselines (#3435)
# ---------------------------------------------------------------------------


def test_shipped_answerability_samples_use_the_gates_own_builder() -> None:
    from kestrel_sovereign.storage.memory_answerability import answerability_decision_requests

    files = ev.sample_files([ev.PACKAGED_SAMPLES_DIR / "memory_answerability"])
    samples, _ = ev.load_samples(files)
    assert len(samples) >= 30
    labels = [v for s in samples for v in s.expected.values()]
    assert any(labels) and not all(labels)
    for sample in samples:
        requests, keys = answerability_decision_requests(
            sample.raw["question"], sample.raw["candidates"])
        assert sample.requests == requests and sample.threshold_keys == keys


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"adapter": "nope", "id": "x"}, "unknown sample adapter"),
        ({"adapter": "memory_answerability", "id": "x", "question": "q",
          "candidates": [], "answerable": []}, "candidates must be"),
        ({"adapter": "memory_answerability", "id": "x", "question": "q",
          "candidates": ["a"], "answerable": [1]}, "distinct candidate indices"),
        ({"adapter": "memory_answerability", "id": "x", "question": " ",
          "candidates": ["a"], "answerable": []}, "question must be"),
    ],
)
def test_adapter_samples_are_validated(raw, message) -> None:
    with pytest.raises(ev.SampleError, match=message):
        ev.parse_sample(json.dumps(raw), "f:1")


@pytest.mark.asyncio
async def test_chat_baseline_is_scored_but_never_proposes(monkeypatch) -> None:
    from kestrel_sovereign.storage import memory_answerability as ma

    raw = {"adapter": "memory_answerability", "id": "p", "question": "pet?",
           "candidates": ["Quasar the axolotl", "gravel"], "answerable": [0]}
    bad = dict(raw, id="q")
    samples = [ev.parse_sample(json.dumps(raw), "f:1"), ev.parse_sample(json.dumps(bad), "f:2")]

    class _Gate:
        def __init__(self, service, **kwargs):
            self.kwargs = kwargs

        async def filter(self, query, candidates):
            if query and candidates[1].content == "gravel" and self.kwargs["force_local_only_provider"]():
                return ma.AnswerabilityDecision(frozenset({"c0"}), True, 1.0)
            return ma.AnswerabilityDecision(frozenset(), False, 1.0, reason="x")

    monkeypatch.setattr(ma, "LLMAnswerabilityGate", _Gate)
    report = await ev.evaluate_baseline(
        object(), "memory_answerability", "chat", samples, local_only=True,
        timeout_seconds=5, concurrency=1, target_accuracy=0.9)
    assert report.baseline and report.selector == "baseline:chat"
    [m] = report.metrics
    assert m.key == "answers" and m.accuracy == 1.0 and m.n == 4
    assert report.proposal_blocker() == "baseline (comparison only)"
    assert ev.render_threshold_snippet([report], caller="memory_answerability",
                                       samples=2, sample_hash="ef" * 32) == ""

    with pytest.raises(ev.SampleError, match="no baseline"):
        await ev.evaluate_baseline(object(), "memory_answerability", "nope", samples,
                                   local_only=True, timeout_seconds=5, concurrency=1,
                                   target_accuracy=0.9)


def test_exact_thresholds_are_not_rounded_into_the_snippet() -> None:
    report = ev.ModelReport("a:b/m", "a:b", "m", expected_keys=["answers"],
                            metrics=[ev.KeyMetrics("answers", "noul", 2, 1.0, 0.0, 0.0, 0.90004)])
    assert '"answers" = 0.90004' in ev.render_threshold_snippet(
        [report], caller="c", samples=2, sample_hash="ab" * 32)


def test_unavailable_errors_carry_their_rejection_reason() -> None:
    from kestrel_sdk.llm.decisions import (
        DecisionUnavailable,
        RejectionReason,
        RouteRejection,
        UnavailableReason,
    )

    error = DecisionUnavailable(
        UnavailableReason.NO_CANDIDATE, "x",
        rejections=[RouteRejection("openrouter:api", RejectionReason.NO_FIT, "m",
                                   "context limit unknown")])
    assert ev._error_label(error) == "DecisionUnavailable(no_fit: context limit unknown)"
    assert ev._error_label(DecisionTransportError("x")) == "DecisionTransportError"


@pytest.mark.asyncio
async def test_cloud_baselines_warm_model_discovery_first(monkeypatch) -> None:
    calls: List[str] = []

    class _Service(_CliService):
        async def discover_all_models(self, *a, **k):
            calls.append("discover")
            return []

        async def _ensure_models_discovered(self, *, force_local_only=False):
            calls.append(f"discover_local_only={force_local_only}")

    async def fake_eval(service, selector, samples, **kwargs):
        return ev.ModelReport(selector, *selector.split("/", 1))

    async def fake_baseline(service, caller, name, samples, **kwargs):
        calls.append(f"baseline:{name}")
        return ev.ModelReport(f"baseline:{name}", "baseline", name, baseline=True)

    monkeypatch.setattr(cli_decisions, "evaluate_model", fake_eval)
    monkeypatch.setattr(cli_decisions, "evaluate_baseline", fake_baseline)
    args = argparse.Namespace(caller="memory_answerability", samples=None, route=None, model=None,
                              local_only=False, allow_cloud=False, timeout=5.0, concurrency=1,
                              target_accuracy=0.9, json=None, baseline=["chat"])
    assert await cli_decisions._eval(_Service(), args) == 0
    assert calls[:2] == ["discover", "baseline:chat"]

    # #3491: a local-only chat-audit baseline now runs on local routes, so
    # they are warmed without contacting a cloud vendor.
    calls.clear()
    args.local_only = True
    assert await cli_decisions._eval(_Service(), args) == 0
    assert "discover" not in calls
    assert calls[:2] == ["discover_local_only=True", "baseline:chat"]
