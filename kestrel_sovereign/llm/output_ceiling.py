"""A model's output ceiling, and what a response that reaches it looks like.

Two halves of one fact (#3300):

* **The ceiling.** How many tokens a model may emit in one response is a
  property of the model, reported by its provider (Anthropic's Models API
  publishes it as ``max_tokens``, Gemini's as ``outputTokenLimit``,
  OpenRouter's catalog as ``top_provider.max_completion_tokens``), and carried
  on :attr:`~kestrel_sovereign.llm.model_metadata.ModelInfo.output_limit`. It
  is never a framework literal: the Anthropic adapter used to send
  ``max_tokens: 4096`` on every request (#3300) and the Gemini adapter
  ``max_output_tokens: 8192`` (#3355), so every turn was cut there whatever the
  model allowed. :class:`OutputCeilings` remembers what each route's provider
  reported. When the provider cannot say what a model's ceiling is,
  :class:`OutputCeilingUnknownError` names that instead of a number being
  guessed.

* **The cut.** A response that stops *because* it reached the ceiling is
  incomplete. The provider says so in its stop reason, which an adapter records
  on the response with :func:`attach_stop_reason`, and which the LLM service
  copies into the call's telemetry (``llm_calls`` metadata and the
  ``llm.usage`` line). When the ceiling was the model's own maximum — not a
  smaller budget a caller chose — the adapter also appends
  :func:`output_ceiling_notice` to the response text, so the cut is visible to
  the person reading it, in the persisted conversation row, and to the model on
  the next turn. :func:`output_limit_cut_notice` makes that decision for every
  adapter. Previously a cut turn was indistinguishable from a finished one: a
  233-second turn returned HTTP 200 with an empty body.

* **The cap.** A local route has no server-side output limit of its own, so a
  small model stuck in a repetition loop generated 129,966 tokens over 19
  minutes while holding the turn lock (#3552). Such a route sends an output
  cap on every chat call, and a response that stops on that cap is not an
  answer at all: the adapter raises :class:`OutputCapReachedError`, an
  :class:`IncompleteGenerationError`, and the attempt fails.
  :func:`incomplete_generation` finds one inside the service's wrappers so a
  transport can say what happened without reflecting any error text.

Whether a generation may become a turn's answer at all is decided in one
place, :func:`kestrel_sovereign.llm.generation_gate.judge_generation`, from
the stop reason :func:`attach_stop_reason` records.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

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
            f"{provider} reported no output ceiling for model {model!r}; "
            f"refusing to guess one. Pass max_tokens explicitly, or use a "
            f"model the provider's model metadata describes."
        )


class IncompleteGenerationError(RuntimeError):
    """A model call ended without a finished response.

    Raised instead of returning what the model produced, so the turn sees a
    failed attempt rather than a cut-off answer presented as complete (#3552).
    """

    #: Why the call ended, short and content-free: the detail on the failed
    #: call's card in the chat.
    summary = "did not finish"


class OutputCapReachedError(IncompleteGenerationError):
    """A response stopped at its output cap, not because it finished.

    ``cap`` is the budget the request carried: the route's cap, or a smaller
    one the call asked for. ``route_cap`` is the route's own cap (``cap`` when
    not given). ``cap`` is ``None`` when only the provider's stop reason
    (``stop_reason``) says the response hit an output limit, not how large.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        cap: Optional[int],
        route_cap: Optional[int] = None,
        stop_reason: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.cap = cap
        self.route_cap = cap if route_cap is None else route_cap
        self.stop_reason = stop_reason
        if cap is None:
            self.summary = "stopped at its output limit"
            super().__init__(
                f"{provider} stopped model {model!r} at an output limit "
                f"(stop_reason={stop_reason}) before it finished; the attempt "
                f"failed."
            )
            return
        self.summary = f"stopped at the output cap of {cap:,} tokens"
        if self.route_cap > cap:
            remedy = (
                f"The call asked for at most {cap:,} tokens; the route's cap is "
                f"{self.route_cap:,}."
            )
        else:
            remedy = (
                "Raise the route's max_output_tokens if this model needs longer "
                "answers."
            )
        super().__init__(
            f"{provider} stopped model {model!r} at its output cap of {cap:,} "
            f"tokens before it finished; the attempt failed. {remedy}"
        )


#: Runtime attribute on an aggregate of several routes' errors: its
#: incomplete-generation verdict (see
#: :func:`~kestrel_sovereign.llm.generation_gate.first_unfinished_attempt`).
INCOMPLETE_ATTEMPT_ATTR = "incomplete_attempt"


def incomplete_generation(
    error: BaseException,
) -> Optional[IncompleteGenerationError]:
    """The :class:`IncompleteGenerationError` ``error`` is or wraps, if any.

    Follows the same explicit links as
    :func:`~kestrel_sovereign.llm.retry.advised_wait_exceeding_budget`:
    ``__cause__``, ``original_error`` and ``underlying``, never the implicit
    ``__context__``. An aggregate of several routes' errors (one carrying
    ``declined_wait``) states its own verdict in
    :data:`INCOMPLETE_ATTEMPT_ATTR`, and its links are not read past: they
    lead to the last route's error, which describes neither the routes before
    it nor the call as a whole.
    """
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, IncompleteGenerationError):
            return current
        if hasattr(current, "declined_wait"):
            verdict = getattr(current, INCOMPLETE_ATTEMPT_ATTR, None)
            return verdict if isinstance(verdict, IncompleteGenerationError) else None
        for link in (
            current.__cause__,
            getattr(current, "original_error", None),
            getattr(current, "underlying", None),
        ):
            if isinstance(link, BaseException):
                pending.append(link)
    return None


def reported_token_limit(value: Any) -> Optional[int]:
    """``value`` when it is a real token limit — a positive integer — else ``None``.

    Providers say "not known" as a missing field, ``null`` or ``0``. All of
    them come back as ``None`` here, so no caller mistakes one for a limit, and
    nobody substitutes a number of their own (#3355: OpenRouter models without
    a ``context_length`` were recorded as having a 4,096-token window).
    """
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


class OutputCeilings:
    """The output ceilings a provider reported, keyed by model id.

    One per adapter instance, so each route keeps what its own credentials
    were told. Filled by model discovery (:meth:`learn`) and by the adapter's
    own per-model lookup for a model discovery did not describe
    (:meth:`remember`).
    """

    def __init__(self) -> None:
        self._by_model: Dict[str, int] = {}

    def learn(self, models: Iterable[Any]) -> None:
        """Remember the ceiling each discovered ``ModelInfo`` reports."""
        for info in models:
            limit = reported_token_limit(getattr(info, "output_limit", None))
            if limit is not None:
                self._by_model[info.id] = limit

    def get(self, model: str) -> Optional[int]:
        """The ceiling reported for ``model``, or ``None`` if none is known."""
        return self._by_model.get(model)

    def remember(self, model: str, limit: int) -> None:
        """Record the ceiling a lookup reported for ``model``."""
        self._by_model[model] = limit


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


def output_ceiling_notice(*, ceiling: Optional[int], stop_reason: str) -> str:
    """The host-authored line marking a response cut at the model's ceiling.

    ``ceiling`` is ``None`` when no budget was sent and the provider applied
    an output limit of its own without saying how large it is.
    """
    if ceiling is None:
        return (
            f"[Output limit reached: the provider stopped this response at its "
            f"output limit (stop_reason={stop_reason}) and did not report how "
            f"many tokens that is. This response is incomplete.]"
        )
    return (
        f"[Output limit reached: the model stopped at its maximum output of "
        f"{ceiling:,} tokens (stop_reason={stop_reason}). This response is "
        f"incomplete.]"
    )


def output_limit_cut_notice(
    *,
    provider: str,
    model: str,
    stop_reason: Optional[str],
    cut_reason: str,
    caller_budget: bool,
    model_ceiling: Optional[int] = None,
) -> Optional[str]:
    """The notice for a response cut at an output limit nobody chose, else None.

    ``cut_reason`` is the provider's own stop reason for "stopped at the output
    limit" (``max_tokens`` for Anthropic, ``MAX_TOKENS`` for Gemini, ``length``
    for OpenAI-compatible APIs). ``caller_budget`` says a caller chose the
    output budget: that is the caller's contract, its response keeps its text
    untouched, and ``stop_reason`` still says where it stopped. Otherwise the
    response could not have been finished at all, and presenting it as an
    answer is the #3300 defect, so it gets the notice. ``model_ceiling`` is the
    model's own ceiling the adapter sent, or ``None`` when the adapter sent no
    budget and the provider applied its own limit.
    """
    if stop_reason != cut_reason:
        return None
    if caller_budget:
        logger.info(
            "%s response for %s stopped at the caller's max_tokens budget "
            "(stop_reason=%s)", provider, model, stop_reason,
        )
        return None
    if model_ceiling is None:
        logger.warning(
            "%s response for %s stopped at the provider's output limit "
            "(stop_reason=%s); the response is incomplete",
            provider, model, stop_reason,
        )
    else:
        logger.warning(
            "%s response for %s reached the model's output ceiling of %d "
            "tokens (stop_reason=%s); the response is incomplete",
            provider, model, model_ceiling, stop_reason,
        )
    return output_ceiling_notice(ceiling=model_ceiling, stop_reason=stop_reason)


def context_window_notice(*, stop_reason: str) -> str:
    """The host-authored line marking a response cut because the whole
    conversation filled the model's context window."""
    return (
        f"[Context window full: the model stopped because this conversation "
        f"reached its context window (stop_reason={stop_reason}). This "
        f"response is incomplete.]"
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
    "INCOMPLETE_ATTEMPT_ATTR",
    "IncompleteGenerationError",
    "OutputCapReachedError",
    "OutputCeilingUnknownError",
    "OutputCeilings",
    "attach_stop_reason",
    "context_window_notice",
    "incomplete_generation",
    "join_output_ceiling_notice",
    "output_ceiling_notice",
    "output_ceiling_notice_chunk",
    "output_limit_cut_notice",
    "reported_token_limit",
    "response_stop_reason",
]
