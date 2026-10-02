"""Whether a validated request fits a decision model's limits (§7).

The request was measured once during validation. The fit check reuses those
bytes; it never re-tokenises and never truncates ``state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kestrel_sdk._frozen_json import canonical_json_bytes
from kestrel_sdk.llm.decisions import (
    ChoiceQuestion,
    DecisionModelInfo,
    ScoreQuestion,
    ValidatedDecisionRequest,
)

from kestrel_sovereign.agent.token_counter import CHARS_PER_TOKEN_ESTIMATE

#: Headroom for the model's own prompt framing around one question (system
#: prompt, field labels, and the answer token positions letter-scoring
#: backends reserve).
FRAMING_TOKENS = 512
#: Allowance for the fields a route's envelope adds around the canonical
#: request (``model``, ``keep_alive``, OpenRouter's ``provider``/``user``).
ENVELOPE_BYTES = 1024


@dataclass(frozen=True)
class RequestFootprint:
    """Sizes of one validated request, measured once."""

    total_bytes: int
    state_bytes: int
    largest_question_bytes: int
    question_count: int
    largest_option_count: int

    @classmethod
    def of(cls, request: ValidatedDecisionRequest) -> "RequestFootprint":
        state_bytes = len(canonical_json_bytes(request.state))
        questions = request.wire["questions"]
        largest = max(len(canonical_json_bytes(q)) for q in questions.values())  # type: ignore[union-attr]
        option_counts = [
            len(q.options) if isinstance(q, ChoiceQuestion) else len(q.levels)
            for q in request.questions.values()
            if isinstance(q, (ChoiceQuestion, ScoreQuestion))
        ]
        return cls(
            total_bytes=request.encoded_bytes,
            state_bytes=state_bytes,
            largest_question_bytes=largest,
            question_count=len(request.questions),
            largest_option_count=max(option_counts, default=0),
        )

    @property
    def per_question_tokens(self) -> int:
        """Estimated tokens one question's evaluation sees: state + question.

        Every route evaluates each question against the whole state (serial
        backends literally re-send it), so this is what must fit the window.
        """

        size = self.state_bytes + self.largest_question_bytes
        return -(-size // CHARS_PER_TOKEN_ESTIMATE) + FRAMING_TOKENS


def misfit(
    info: DecisionModelInfo,
    footprint: RequestFootprint,
    *,
    context_limit_override: Optional[int] = None,
) -> Optional[str]:
    """Return why the request does not fit ``info``, or ``None`` if it fits."""

    limit = context_limit_override if context_limit_override is not None else info.context_limit
    if limit is None:
        return "context limit unknown"
    if footprint.per_question_tokens > limit:
        return (
            f"needs ~{footprint.per_question_tokens} tokens per question, "
            f"context limit is {limit}"
        )
    envelope_bytes = footprint.total_bytes + ENVELOPE_BYTES
    if info.max_request_bytes is not None and envelope_bytes > info.max_request_bytes:
        return (
            f"request is ~{envelope_bytes} bytes with its envelope, route "
            f"accepts {info.max_request_bytes}"
        )
    if info.max_questions is not None and footprint.question_count > info.max_questions:
        return (
            f"{footprint.question_count} questions, route accepts {info.max_questions}"
        )
    if info.max_options is not None and footprint.largest_option_count > info.max_options:
        return (
            f"a question has {footprint.largest_option_count} options, route "
            f"accepts {info.max_options}"
        )
    return None
