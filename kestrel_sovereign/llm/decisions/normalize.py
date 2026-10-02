"""Turn a route's raw systemone response into a checked answer set (§2.3).

Kestrel owns the meaning of an answer. The normaliser enforces exact coverage,
numeric sanity, recomputes the argmax and the score from the distribution, and
drops the vendor's ``confidence`` entirely: vendors define it differently, so
a caller thresholding on it would change behaviour when the route changes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional

from kestrel_sdk.llm.decisions import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionProtocolError,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    ValidatedDecisionRequest,
)

logger = logging.getLogger(__name__)

#: How far a distribution may sum from 1 before it is refused instead of
#: renormalised.
SUM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class NormalizedResponse:
    answers: Mapping[str, Answer]
    input_tokens: Optional[int]
    cost_usd: Optional[float]
    usage_available: bool


def _probability(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionProtocolError(f"{where} is not a number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise DecisionProtocolError(f"{where} is outside [0, 1]: {value!r}")
    return number


def _distribution(values: dict[Any, float], where: str) -> dict[Any, float]:
    total = sum(values.values())
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise DecisionProtocolError(f"{where} sums to {total:.6f}, not 1")
    return {key: value / total for key, value in values.items()}


def _choice(question: ChoiceQuestion, raw: Mapping[str, Any], where: str) -> ChoiceAnswer:
    probabilities = raw.get("probabilities")
    if not isinstance(probabilities, Mapping):
        raise DecisionProtocolError(f"{where} has no probabilities object")
    expected = set(question.options)
    if set(probabilities) != expected:
        raise DecisionProtocolError(
            f"{where} probabilities cover {sorted(probabilities)!r}, "
            f"expected exactly {sorted(expected)!r}"
        )
    distribution = _distribution(
        {key: _probability(value, f"{where} p({key})") for key, value in probabilities.items()},
        where,
    )
    # Ties break in the caller's declared option order.
    choice = max(question.options, key=lambda option: distribution[option])
    reported = raw.get("choice")
    if reported is not None and reported != choice:
        logger.warning(
            "decision %s: vendor choice %r disagrees with argmax %r; using argmax",
            where,
            reported,
            choice,
        )
    ordered = {option: distribution[option] for option in question.options}
    return ChoiceAnswer(choice=choice, probabilities=MappingProxyType(ordered))


def _score(question: ScoreQuestion, raw: Mapping[str, Any], where: str) -> ScoreAnswer:
    probabilities = raw.get("probabilities")
    if not isinstance(probabilities, Mapping):
        raise DecisionProtocolError(f"{where} has no probabilities object")
    expected = {str(index) for index in range(len(question.levels))}
    if set(probabilities) != expected:
        raise DecisionProtocolError(
            f"{where} probabilities cover levels {sorted(probabilities)!r}, "
            f"expected exactly {sorted(expected, key=int)!r}"
        )
    distribution = _distribution(
        {
            int(level): _probability(value, f"{where} p(level {level})")
            for level, value in probabilities.items()
        },
        where,
    )
    ordered = tuple(distribution[index] for index in range(len(question.levels)))
    score = sum(index * p for index, p in enumerate(ordered))
    return ScoreAnswer(score=score, probabilities=ordered)


def _noul(raw: Mapping[str, Any], where: str) -> NoulAnswer:
    if "noul" not in raw:
        raise DecisionProtocolError(f"{where} has no noul value")
    return NoulAnswer(p_true=_probability(raw["noul"], f"{where} noul"))


def _usage(body: Mapping[str, Any]) -> tuple[Optional[int], Optional[float], bool]:
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        return None, None, False
    tokens = usage.get("input_tokens")
    input_tokens = (
        int(tokens)
        if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0
        else None
    )
    cost = usage.get("cost")
    cost_usd = (
        float(cost)
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost)
        else None
    )
    return input_tokens, cost_usd, input_tokens is not None


def normalize_response(
    request: ValidatedDecisionRequest, body: Any
) -> NormalizedResponse:
    """Check ``body`` against ``request`` and return Kestrel's answer set.

    Any coverage, type or numeric violation raises
    :class:`DecisionProtocolError` and refuses the whole result.
    """

    if not isinstance(body, Mapping):
        raise DecisionProtocolError("decision response is not a JSON object")
    raw_answers = body.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise DecisionProtocolError("decision response has no answers object")
    expected_ids = set(request.questions)
    if set(raw_answers) != expected_ids:
        missing = sorted(expected_ids - set(raw_answers))
        extra = sorted(set(raw_answers) - expected_ids)
        raise DecisionProtocolError(
            f"decision response answers mismatch: missing {missing!r}, extra {extra!r}"
        )

    answers: dict[str, Answer] = {}
    for question_id, question in request.questions.items():
        raw = raw_answers[question_id]
        where = f"answer {question_id!r}"
        if not isinstance(raw, Mapping):
            raise DecisionProtocolError(f"{where} is not an object")
        if isinstance(question, ChoiceQuestion):
            expected_type, answer = "choice", _choice
        elif isinstance(question, ScoreQuestion):
            expected_type, answer = "score", _score
        else:
            assert isinstance(question, NoulQuestion)
            expected_type, answer = "noul", None
        reported_type = raw.get("type")
        if reported_type is not None and reported_type != expected_type:
            raise DecisionProtocolError(
                f"{where} has type {reported_type!r}, expected {expected_type!r}"
            )
        answers[question_id] = (
            _noul(raw, where) if answer is None else answer(question, raw, where)  # type: ignore[arg-type]
        )

    input_tokens, cost_usd, usage_available = _usage(body)
    return NormalizedResponse(
        answers=MappingProxyType(answers),
        input_tokens=input_tokens,
        cost_usd=cost_usd,
        usage_available=usage_available,
    )
