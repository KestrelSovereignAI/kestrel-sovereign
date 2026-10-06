"""Eval harness for decision models (spec §9).

Labelled samples are run through ``LLMService.decide`` against each candidate
decision model, and the answers are scored per threshold key: accuracy, Brier
score, expected calibration error, latency, and a proposed threshold. The
proposal is printed as a ``[decisions.thresholds...]`` snippet carrying the
sample-set hash and date, so every configured threshold traces back to the
evidence that set it.

Sample format (JSON Lines, one sample per line)::

    {"id": "pet-name",
     "state": {...},
     "questions": {"c0": {"type": "noul", "instructions": "...",
                          "criteria": {"true": "...", "false": "..."}}},
     "expected": {"c0": true},
     "threshold_keys": {"c0": "answers"}}

``questions`` use the systemone wire shape. ``expected`` is a bool for
``noul``, an option id for ``choice`` and a level index for ``score``.
``threshold_keys`` is optional, as in ``decide``.

A caller may instead register a sample *adapter*: a line carrying
``"adapter": "<caller>"`` is built by that caller's own request builder, so
the eval measures exactly the request the caller sends. A caller may also
register *baselines* (``--baseline``): another way of making the same
judgement, scored on the same samples for comparison.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from kestrel_sdk.llm.decisions import (
    ChoiceQuestion,
    DecisionError,
    DecisionRequest,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreQuestion,
)

#: Package-shipped sample sets: ``eval_samples/<caller>/*.jsonl``.
PACKAGED_SAMPLES_DIR = Path(__file__).with_name("eval_samples")

#: Bins for expected calibration error.
ECE_BINS = 10

#: Caller-owned sample adapters: ``adapter`` name -> ``"module:function"``
#: taking ``(raw, source)`` and returning a :class:`Sample`.
SAMPLE_ADAPTERS: Mapping[str, str] = {
    "memory_answerability": (
        "kestrel_sovereign.storage.memory_answerability:answerability_eval_sample"
    ),
    "memory_attestation": (
        "kestrel_sovereign.features.memory.reflection_hook:attestation_eval_sample"
    ),
    "response_audit": (
        "kestrel_sovereign.features.response_audit.decision_audit:response_audit_eval_sample"
    ),
}

#: Caller-owned baselines: caller -> {name: "module:function"}. A baseline is
#: ``async (llm_service, sample, *, timeout_seconds, local_only)`` returning
#: ``{question_id: bool}`` for noul samples, or ``None`` when it failed.
BASELINES: Mapping[str, Mapping[str, str]] = {
    "memory_answerability": {
        "chat": "kestrel_sovereign.storage.memory_answerability:answerability_chat_baseline",
    },
    "memory_attestation": {
        "chat": "kestrel_sovereign.features.memory.reflection_hook:attestation_chat_baseline",
    },
    "response_audit": {
        "chat": (
            "kestrel_sovereign.features.response_audit.decision_audit:response_audit_chat_baseline"
        ),
    },
}


def _resolve(target: str) -> Any:
    module, _, name = target.partition(":")
    return getattr(importlib.import_module(module), name)


class SampleError(ValueError):
    """A sample file is malformed. The message names the file and line."""


@dataclass(frozen=True)
class Sample:
    id: str
    request: DecisionRequest
    expected: Mapping[str, Any]
    threshold_keys: Mapping[str, str]
    source: str
    #: The adapter's original fields, for caller-owned baselines.
    raw: Optional[Mapping[str, Any]] = None


def _question(raw: Any, where: str) -> Question:
    if not isinstance(raw, Mapping):
        raise SampleError(f"{where}: question must be an object")
    kind = raw.get("type")
    instructions = raw.get("instructions")
    criteria = raw.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, Mapping):
            raise SampleError(f"{where}: choice criteria must be an object")
        return ChoiceQuestion(instructions=instructions, options=dict(criteria))
    if kind == "score":
        if not isinstance(criteria, list):
            raise SampleError(f"{where}: score criteria must be a list of levels")
        return ScoreQuestion(instructions=instructions, levels=list(criteria))
    if kind == "noul":
        criteria = criteria or {}
        if not isinstance(criteria, Mapping):
            raise SampleError(f"{where}: noul criteria must be an object")
        return NoulQuestion(
            instructions=instructions,
            true_means=criteria.get("true"),
            false_means=criteria.get("false"),
        )
    raise SampleError(f"{where}: unknown question type {kind!r}")


def _expected(question: Question, value: Any, where: str) -> Any:
    if isinstance(question, NoulQuestion):
        if not isinstance(value, bool):
            raise SampleError(f"{where}: noul expectation must be true/false")
    elif isinstance(question, ChoiceQuestion):
        if value not in question.options:
            raise SampleError(f"{where}: choice expectation {value!r} is not an option")
    elif isinstance(question, ScoreQuestion):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(question.levels):
            raise SampleError(f"{where}: score expectation must be a level index")
    return value


def parse_sample(line: str, source: str) -> Sample:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as error:
        raise SampleError(f"{source}: invalid JSON ({error.msg})") from None
    if not isinstance(raw, Mapping):
        raise SampleError(f"{source}: sample must be an object")
    adapter = raw.get("adapter")
    if adapter is not None:
        target = SAMPLE_ADAPTERS.get(adapter) if isinstance(adapter, str) else None
        if target is None:
            raise SampleError(f"{source}: unknown sample adapter {adapter!r}")
        return _resolve(target)(raw, source)
    sample_id = raw.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise SampleError(f"{source}: sample needs a non-empty string id")
    where = f"{source} [{sample_id}]"
    raw_questions = raw.get("questions")
    if not isinstance(raw_questions, Mapping) or not raw_questions:
        raise SampleError(f"{where}: questions must be a non-empty object")
    questions = {qid: _question(q, f"{where} {qid}") for qid, q in raw_questions.items()}
    raw_expected = raw.get("expected")
    if not isinstance(raw_expected, Mapping) or set(raw_expected) != set(questions):
        raise SampleError(f"{where}: expected must label exactly the questions asked")
    expected = {qid: _expected(questions[qid], raw_expected[qid], f"{where} {qid}") for qid in questions}
    keys = raw.get("threshold_keys") or {}
    if not isinstance(keys, Mapping):
        raise SampleError(f"{where}: threshold_keys must be an object")
    return Sample(
        id=sample_id,
        request=DecisionRequest(state=raw.get("state"), questions=questions),
        expected=expected,
        threshold_keys=dict(keys),
        source=source,
    )


def sample_files(paths: Iterable[Path]) -> List[Path]:
    """Expand files and directories into a sorted list of ``*.jsonl`` files."""

    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            raise SampleError(f"{path}: no such sample file or directory")
    return sorted(set(files))


def load_samples(files: Sequence[Path]) -> Tuple[List[Sample], str]:
    """Load samples and return them with the sample set's content hash."""

    digest = hashlib.sha256()
    samples: List[Sample] = []
    seen: set[str] = set()
    for path in files:
        content = path.read_bytes()
        digest.update(path.name.encode("utf-8") + b"\0" + content + b"\0")
        for number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            sample = parse_sample(line, f"{path.name}:{number}")
            if sample.id in seen:
                raise SampleError(f"{path.name}:{number}: duplicate sample id {sample.id!r}")
            seen.add(sample.id)
            samples.append(sample)
    if not samples:
        raise SampleError("no samples found")
    return samples, digest.hexdigest()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    key: str
    question: Question
    expected: Any
    answer: Any


@dataclass
class KeyMetrics:
    key: str
    kind: str
    n: int
    accuracy: float
    brier: float
    ece: float
    proposed_threshold: Optional[float]
    note: str = ""
    extra: Dict[str, float] = field(default_factory=dict)


def _ece(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    bins: Dict[int, List[Tuple[float, bool]]] = {}
    for confidence, hit in zip(confidences, correct):
        index = min(int(confidence * ECE_BINS), ECE_BINS - 1)
        bins.setdefault(index, []).append((confidence, hit))
    total = len(confidences)
    return sum(
        len(members) / total
        * abs(statistics.fmean(c for c, _ in members) - statistics.fmean(1.0 if h else 0.0 for _, h in members))
        for members in bins.values()
    )


def _noul_metrics(key: str, observations: Sequence[Observation]) -> KeyMetrics:
    probs = [o.answer.p_true for o in observations]
    labels = [bool(o.expected) for o in observations]
    brier = statistics.fmean((p - (1.0 if y else 0.0)) ** 2 for p, y in zip(probs, labels))
    ece = _ece(probs, labels)
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        accuracy = statistics.fmean((p >= 0.5) == y for p, y in zip(probs, labels))
        return KeyMetrics(key, "noul", len(labels), accuracy, brier, ece, None,
                          note="needs both true and false samples to propose a threshold")

    # Youden's J over every distinct cut point; ties go to the cut nearest 0.5.
    cuts = sorted(set(probs) | {0.5})
    best: Tuple[float, float, float] = (-2.0, 0.0, 0.5)
    for cut in cuts:
        tpr = sum(p >= cut and y for p, y in zip(probs, labels)) / positives
        fpr = sum(p >= cut and not y for p, y in zip(probs, labels)) / negatives
        candidate = (tpr - fpr, -abs(cut - 0.5), cut)
        if candidate > best:
            best = candidate
    threshold = best[2]
    predicted = [p >= threshold for p in probs]
    tp = sum(p and y for p, y in zip(predicted, labels))
    fp = sum(p and not y for p, y in zip(predicted, labels))
    fn = sum(not p and y for p, y in zip(predicted, labels))
    accuracy = statistics.fmean(p == y for p, y in zip(predicted, labels))
    return KeyMetrics(
        key, "noul", len(labels), accuracy, brier, ece, threshold,
        extra={
            "precision": tp / (tp + fp) if tp + fp else math.nan,
            "recall": tp / (tp + fn) if tp + fn else math.nan,
            "accuracy@0.5": statistics.fmean((p >= 0.5) == y for p, y in zip(probs, labels)),
        },
    )


def _categorical(o: Observation) -> Tuple[List[float], int, int]:
    if isinstance(o.question, ChoiceQuestion):
        options = list(o.question.options)
        probs = [o.answer.probabilities[opt] for opt in options]
        return probs, options.index(o.expected), options.index(o.answer.choice)
    probs = list(o.answer.probabilities)
    return probs, int(o.expected), max(range(len(probs)), key=probs.__getitem__)


def _categorical_metrics(
    key: str, kind: str, observations: Sequence[Observation], target_accuracy: float
) -> KeyMetrics:
    rows = [_categorical(o) for o in observations]
    correct = [pred == truth for _, truth, pred in rows]
    top = [max(probs) for probs, _, _ in rows]
    brier = statistics.fmean(
        sum((p - (1.0 if i == truth else 0.0)) ** 2 for i, p in enumerate(probs))
        for probs, truth, _ in rows
    )
    ece = _ece(top, correct)
    accuracy = statistics.fmean(correct)

    # Smallest top-probability cut at which the answers the caller would act
    # on reach the target accuracy.
    threshold: Optional[float] = None
    coverage = 0.0
    for cut in sorted(set(top)):
        kept = [hit for confidence, hit in zip(top, correct) if confidence >= cut]
        if kept and statistics.fmean(kept) >= target_accuracy:
            threshold, coverage = cut, len(kept) / len(top)
            break
    extra: Dict[str, float] = {"coverage": coverage}
    if kind == "score":
        extra["mae"] = statistics.fmean(
            abs(o.answer.score - int(o.expected)) for o in observations
        )
    return KeyMetrics(
        key, kind, len(rows), accuracy, brier, ece, threshold,
        note="" if threshold is not None else f"no cut reaches accuracy {target_accuracy}",
        extra=extra,
    )


def score_observations(
    observations: Sequence[Observation], *, target_accuracy: float = 0.9
) -> List[KeyMetrics]:
    by_key: Dict[str, List[Observation]] = {}
    for observation in observations:
        by_key.setdefault(observation.key, []).append(observation)
    metrics: List[KeyMetrics] = []
    for key, group in sorted(by_key.items()):
        kinds = {type(o.question) for o in group}
        if len(kinds) != 1:
            raise SampleError(f"threshold key {key!r} mixes question types")
        if isinstance(group[0].question, NoulQuestion):
            metrics.append(_noul_metrics(key, group))
        else:
            kind = "choice" if isinstance(group[0].question, ChoiceQuestion) else "score"
            metrics.append(_categorical_metrics(key, kind, group, target_accuracy))
    return metrics


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class ModelReport:
    selector: str
    route: str
    model: str
    metrics: List[KeyMetrics] = field(default_factory=list)
    latencies_ms: List[int] = field(default_factory=list)
    errors: Dict[str, int] = field(default_factory=dict)
    expected_keys: List[str] = field(default_factory=list)
    baseline: bool = False

    def proposal_blocker(self) -> Optional[str]:
        """Why this run cannot back a calibration, or ``None`` if it can.

        A proposal must come from a complete run: every sample answered and
        every threshold key in the sample set covered with a threshold.
        Anything less is biased by the missing labels (§2.5, §9).
        """

        if self.baseline:
            return "baseline (comparison only)"
        if self.errors:
            return "some samples failed"
        if not self.metrics:
            return "no answers"
        covered = {m.key for m in self.metrics if m.proposed_threshold is not None}
        missing = sorted((set(self.expected_keys) | {m.key for m in self.metrics}) - covered)
        if missing:
            return f"no threshold for key(s) {', '.join(missing)}"
        return None

    def latency(self, quantile: float) -> Optional[float]:
        if not self.latencies_ms:
            return None
        ordered = sorted(self.latencies_ms)
        index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
        return float(ordered[index])


def _error_label(error: DecisionError) -> str:
    """Error class, plus the per-route reason for an unavailable model."""

    name = type(error).__name__
    reasons = sorted({r.reason.value for r in getattr(error, "rejections", ())})
    detail = sorted({r.detail for r in getattr(error, "rejections", ()) if r.detail})
    if reasons:
        name += f"({', '.join(reasons)}" + (f": {'; '.join(detail)}" if detail else "") + ")"
    return name


def eval_caller_id(caller: str) -> str:
    """Eval traffic is recorded under its own caller, never the real one.

    It also keeps the real caller's ``uncalibrated = "refuse"`` policy from
    hiding exactly the models an eval exists to calibrate.
    """

    return f"kestrel.eval.{caller}"


async def evaluate_model(
    llm_service: Any,
    selector: str,
    samples: Sequence[Sample],
    *,
    caller: str,
    local_only: bool,
    timeout_seconds: float,
    concurrency: int,
    target_accuracy: float,
) -> ModelReport:
    """Run every sample against one model (``<vendor>:<route>/<model>``)."""

    route, _, model = selector.partition("/")
    expected_keys = sorted({
        sample.threshold_keys.get(qid, qid)
        for sample in samples
        for qid in sample.request.questions
    })
    report = ModelReport(selector=selector, route=route, model=model, expected_keys=expected_keys)
    gate = asyncio.Semaphore(max(1, concurrency))
    observations: List[Observation] = []

    async def one(sample: Sample) -> None:
        async with gate:
            try:
                result = await llm_service.decide(
                    sample.request,
                    caller=eval_caller_id(caller),
                    timeout_seconds=timeout_seconds,
                    model_override=selector,
                    local_only=local_only,
                    threshold_keys=sample.threshold_keys or None,
                )
            except DecisionError as error:
                name = _error_label(error)
                report.errors[name] = report.errors.get(name, 0) + 1
                return
        report.latencies_ms.append(result.duration_ms)
        for qid, question in sample.request.questions.items():
            observations.append(Observation(
                key=sample.threshold_keys.get(qid, qid),
                question=question,
                expected=sample.expected[qid],
                answer=result.answers[qid],
            ))

    await asyncio.gather(*(one(sample) for sample in samples))
    if observations:
        report.metrics = score_observations(observations, target_accuracy=target_accuracy)
    return report


async def evaluate_baseline(
    llm_service: Any,
    caller: str,
    name: str,
    samples: Sequence[Sample],
    *,
    local_only: bool,
    timeout_seconds: float,
    concurrency: int,
    target_accuracy: float,
) -> ModelReport:
    """Score a caller-registered baseline on the same samples.

    Its verdicts are scored as probabilities 1.0 / 0.0, so accuracy,
    precision and recall line up with the decision models'; it never proposes
    a calibration.
    """

    target = (BASELINES.get(caller) or {}).get(name)
    if target is None:
        raise SampleError(f"caller {caller!r} has no baseline named {name!r}")
    baseline = _resolve(target)
    report = ModelReport(
        selector=f"baseline:{name}", route="baseline", model=name, baseline=True,
        expected_keys=sorted({s.threshold_keys.get(q, q) for s in samples for q in s.request.questions}),
    )
    gate = asyncio.Semaphore(max(1, concurrency))
    observations: List[Observation] = []
    loop = asyncio.get_running_loop()

    async def one(sample: Sample) -> None:
        async with gate:
            started = loop.time()
            verdicts = await baseline(
                llm_service, sample, timeout_seconds=timeout_seconds, local_only=local_only
            )
            elapsed = int((loop.time() - started) * 1000)
        if verdicts is None:
            report.errors["BaselineIncomplete"] = report.errors.get("BaselineIncomplete", 0) + 1
            return
        report.latencies_ms.append(elapsed)
        for qid, question in sample.request.questions.items():
            if not isinstance(question, NoulQuestion):
                raise SampleError(f"baseline {name!r} only scores noul questions")
            observations.append(Observation(
                key=sample.threshold_keys.get(qid, qid),
                question=question,
                expected=sample.expected[qid],
                answer=NoulAnswer(p_true=1.0 if verdicts[qid] else 0.0),
            ))

    await asyncio.gather(*(one(sample) for sample in samples))
    if observations:
        report.metrics = score_observations(observations, target_accuracy=target_accuracy)
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _toml_key(key: str) -> str:
    """A TOML quoted key. JSON string escaping is valid TOML basic-string syntax."""

    return json.dumps(key, ensure_ascii=False)


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def render_report(reports: Sequence[ModelReport], *, samples: int, sample_hash: str) -> str:
    lines = [f"samples: {samples}  set sha256:{sample_hash[:12]}", ""]
    for report in reports:
        errors = ", ".join(f"{k}={v}" for k, v in sorted(report.errors.items())) or "none"
        lines.append(
            f"{report.selector}  answered={len(report.latencies_ms)}/{samples}  "
            f"p50={_fmt(report.latency(0.5), 0)}ms  p95={_fmt(report.latency(0.95), 0)}ms  "
            f"errors: {errors}"
        )
        blocker = report.proposal_blocker()
        if blocker:
            lines.append(f"  no calibration proposal: {blocker}")
        for m in report.metrics:
            extra = "  ".join(f"{k}={_fmt(v)}" for k, v in sorted(m.extra.items()))
            lines.append(
                f"  {m.key:<20} {m.kind:<6} n={m.n:<4} acc={_fmt(m.accuracy)}  "
                f"brier={_fmt(m.brier)}  ece={_fmt(m.ece)}  "
                f"threshold={_fmt(m.proposed_threshold)}  {extra}"
                + (f"  ({m.note})" if m.note else "")
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_threshold_snippet(
    reports: Sequence[ModelReport], *, caller: str, samples: int, sample_hash: str,
    today: Optional[date] = None,
) -> str:
    """TOML the operator may paste into ``kestrel.toml`` to accept proposals.

    Only complete runs are included (see :meth:`ModelReport.proposal_blocker`),
    so the snippet never declares a partial or biased calibration (§2.5).
    Every key is quoted: threshold keys and caller ids may contain dots, which
    TOML would otherwise read as nested tables.
    """

    stamp = (today or date.today()).isoformat()
    blocks: List[str] = []
    for report in reports:
        if report.proposal_blocker() is not None:
            continue
        # repr() keeps the exact cut: rounding down could admit an answer
        # the proposal was chosen to exclude.
        body = "\n".join(
            f"{_toml_key(m.key)} = {m.proposed_threshold!r}" for m in report.metrics
        )
        blocks.append(
            f"# kestrel decisions eval {stamp}: {samples} samples, set sha256:{sample_hash[:12]}\n"
            f"[decisions.thresholds.{_toml_key(caller)}.models.{_toml_key(report.selector)}]\n{body}"
        )
    if not blocks:
        return ""
    # The caller table's policy makes the proposal complete on its own: a
    # caller with no built-in defaults, under the implicit "default" policy
    # and no default table, would have every request refused (§2.5).
    header = (
        f"[decisions.thresholds.{_toml_key(caller)}]\n"
        "# \"refuse\": only the calibrated models below may answer this caller.\n"
        "# \"default\" lets uncalibrated models answer too, using the caller's\n"
        "# built-in defaults or a `default = { <key> = <threshold> }` table.\n"
        'uncalibrated = "refuse"'
    )
    return header + "\n\n" + "\n\n".join(blocks) + "\n"
