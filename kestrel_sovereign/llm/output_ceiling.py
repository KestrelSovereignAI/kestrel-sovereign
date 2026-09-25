"""A model's output ceiling, and what a response that reaches it looks like.

Two halves of one fact (#3300):

* **The ceiling.** How many tokens a model may emit in one response is a
  property of the model, reported by its provider (Anthropic's Models API
  publishes it as ``max_tokens``), and carried on
  :attr:`~kestrel_sovereign.llm.model_metadata.ModelInfo.output_limit`. It is
  never a framework literal: the Anthropic adapter used to send
  ``max_tokens: 4096`` on every request, so every Claude turn was cut at
  4,096 tokens whatever the model allowed. When the provider cannot say what a
  model's ceiling is, :class:`OutputCeilingUnknownError` names that instead of
  a number being guessed.

* **The cut.** A response that stops *because* it reached the ceiling is
  incomplete. The provider says so in its stop reason, which an adapter records
  on the response with :func:`attach_stop_reason`, and which the LLM service
  copies into the call's telemetry (``llm_calls`` metadata and the
  ``llm.usage`` line). When the ceiling was the model's own maximum — not a
  smaller budget a caller chose — the adapter also appends
  :func:`output_ceiling_notice` to the response text, so the cut is visible to
  the person reading it, in the persisted conversation row, and to the model on
  the next turn. Previously a cut turn was indistinguishable from a finished
  one: a 233-second turn returned HTTP 200 with an empty body.
"""

from __future__ import annotations

from typing import Any, Optional

#: Runtime attribute carrying the provider's stop reason on an
#: ``LLMResponse``. The SDK dataclass has no field for it; adapter-supplied
#: extras ride as attributes (see ``executed_tool_calls``), and every reader
#: goes through :func:`response_stop_reason`.
STOP_REASON_ATTR = "stop_reason"


class OutputCeilingUnknownError(LookupError):
    """The provider did not report an output ceiling for a model.

    Raised instead of sending a guessed ``max_tokens``: a guess either cuts
    answers short (the #3300 defect) or asks for more than the model allows.
    """

    def __init__(self, model: str, provider: str) -> None:
        self.model = model
        self.provider = provider
        super().__init__(
            f"{provider} reported no output ceiling (max_tokens) for model "
            f"{model!r}; refusing to guess one. Pass max_tokens explicitly, or "
            f"use a model the provider's Models API describes."
        )


def attach_stop_reason(response: Any, stop_reason: Optional[str]) -> Any:
    """Record the provider's ``stop_reason`` on ``response`` and return it.

    A missing reason is not recorded, so "the provider did not say" stays
    distinguishable from any reason it did give.
    """
    if isinstance(stop_reason, str) and stop_reason:
        setattr(response, STOP_REASON_ATTR, stop_reason)
    return response


def response_stop_reason(response: Any) -> Optional[str]:
    """The provider stop reason an adapter attached to ``response``, if any."""
    reason = getattr(response, STOP_REASON_ATTR, None)
    return reason if isinstance(reason, str) and reason else None


def output_ceiling_notice(*, ceiling: int, stop_reason: str) -> str:
    """The host-authored line marking a response cut at the model's ceiling."""
    return (
        f"[Output limit reached: the model stopped at its maximum output of "
        f"{ceiling:,} tokens (stop_reason={stop_reason}). This response is "
        f"incomplete.]"
    )


def output_ceiling_notice_chunk(notice: str, *, follows_text: bool) -> str:
    """``notice`` as the chunk that ends a stream: its own paragraph when it
    follows text already sent."""
    return f"\n\n{notice}" if follows_text else notice


def join_output_ceiling_notice(text: Optional[str], notice: str) -> str:
    """``text`` followed by ``notice`` — exactly what a stream that sent
    ``text`` and then :func:`output_ceiling_notice_chunk` would have shown."""
    return (text or "") + output_ceiling_notice_chunk(
        notice, follows_text=bool(text),
    )


__all__ = [
    "STOP_REASON_ATTR",
    "OutputCeilingUnknownError",
    "attach_stop_reason",
    "join_output_ceiling_notice",
    "output_ceiling_notice",
    "output_ceiling_notice_chunk",
    "response_stop_reason",
]
