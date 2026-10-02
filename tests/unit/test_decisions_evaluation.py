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
    assert isinstance(sample.request.questions["n"], NoulQuestion)
    assert sample.request.questions["n"].true_means == "yes"
    assert dict(sample.request.questions["c"].options) == {"a": None, "b": "B"}
    assert list(sample.request.questions["s"].levels) == ["lo", "hi"]
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


def test_report_and_snippet_rendering() -> None:
    full = ev.ModelReport("ollama:local/tev1", "ollama:local", "tev1",
                          metrics=[ev.KeyMetrics("answers", "noul", 4, 1.0, 0.02, 0.05, 0.42)],
                          latencies_ms=[100, 120, 140, 400])
    partial = ev.ModelReport("openrouter:api/x", "openrouter:api", "x",
                             metrics=[ev.KeyMetrics("answers", "noul", 2, 0.5, 0.3, 0.2, None,
                                                    note="needs both")])
    text = ev.render_report([full, partial], samples=4, sample_hash="ab" * 32)
    assert "ollama:local/tev1" in text and "p95=400ms" in text and "needs both" in text

    snippet = ev.render_threshold_snippet([full, partial], caller="memory_answerability",
                                          samples=4, sample_hash="ab" * 32,
                                          today=date(2026, 10, 2))
    assert '[decisions.thresholds.memory_answerability.models."ollama:local/tev1"]' in snippet
    assert "answers = 0.42" in snippet and "2026-10-02" in snippet and "abababababab" in snippet
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
                    target_accuracy=0.9, json=None)
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
