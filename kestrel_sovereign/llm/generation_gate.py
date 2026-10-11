"""The one gate a model's generation passes to become an answer (#3552).

A generation is accepted — recorded as a turn's answer, or acted on as the
tool calls a turn runs — only on positive evidence that it finished: the
provider's own terminal stop reason is a natural end
(:data:`NATURAL_STOP_REASONS`). Everything else is a failed attempt:

* a stop at an output cap or at the model's context window
  (:data:`OUTPUT_LIMIT_STOP_REASONS`);
* any other stop reason, including one this module does not know;
* no stop reason at all: a route that cannot say whether its model finished
  has not shown that it did;
* a call that failed, timed out, went silent past the orchestrator's
  inactivity bound, or was cut short by Stop.

:func:`judge_generation` is that decision, and the only one. The LLM service
applies it to every attempt of the two calls a turn's answer and tool calls
come from, ``stream_with_tool_detection`` and ``generate_with_messages``,
because the provider's evidence is there: a call returns, or a stream ends,
only with a generation the gate accepted. The turn applies it to the
outcomes only the turn sees, a Stop and a call that raised. A failed attempt
contributes nothing to the answer: the turn records none, and a tool batch it
had already completed is recorded as a fixed checkpoint, so the next turn does
not repeat the action.

Before this gate, each path that could turn a model's output into an answer
decided for itself, and six of them let an unfinished generation through: a
follow-up timeout that was swallowed, the HTTP invoke path, a lower caller
budget, request options that removed the cap, an aggregate of failed routes,
and a fallback route whose answer was appended to a capped one.

:data:`ANSWER_PATHS` enumerates every path by which a generation becomes a
turn's answer or a completed-action checkpoint, and how each reaches this
gate. ``tests/unit/test_generation_gate.py`` checks that each one does, and
that every model call the package makes through an LLM service reference is
on a registered path or is declared not to produce an answer, so a new
answer path has to register here. The gate governs answers and checkpoints
only; model calls that produce neither (audits, summaries, extraction) are
outside it (#3573).

An adapter supplies its evidence with
:func:`~kestrel_sovereign.llm.output_ceiling.attach_stop_reason` on the
response it returns, or on the response that ends its stream. Every in-tree
adapter does. A third-party adapter that does not has every turn it serves
fail, by design, until it does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

from .output_ceiling import (
    IncompleteGenerationError,
    OutputCapReachedError,
    incomplete_generation,
    response_stop_reason,
)

#: Terminal stop reasons that are positive evidence a model finished.
NATURAL_STOP_REASONS = frozenset({
    # Anthropic Messages API.
    "end_turn",
    "stop_sequence",
    "tool_use",
    # OpenAI-compatible chat completions (OpenAI, OpenRouter, llama.cpp) and
    # Ollama.
    "stop",
    "tool_calls",
    "function_call",
    # Gemini and Vertex AI ``FinishReason`` names.
    "STOP",
    # Codex app-server ``TurnStatus``.
    "completed",
})

#: Terminal stop reasons for a response cut at an output limit: the request's
#: output budget, the model's own maximum, or its context window.
OUTPUT_LIMIT_STOP_REASONS = frozenset({
    "max_tokens",
    "model_context_window_exceeded",
    "length",
    "MAX_TOKENS",
})


class NoCompletionEvidenceError(IncompleteGenerationError):
    """A route's response carried no stop reason, so whether it finished is
    unknown, and an unknown is not an answer."""

    summary = "gave no evidence it finished"

    def __init__(self, *, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(
            f"{provider} returned a response for model {model!r} without a "
            f"stop reason, so there is no evidence it finished; the attempt "
            f"failed. The route's adapter must report the provider's stop "
            f"reason (attach_stop_reason)."
        )


class UnfinishedStopError(IncompleteGenerationError):
    """A response stopped for a reason that is not a natural end: a safety
    stop, a refusal, a paused or interrupted turn, or a reason this gate does
    not know."""

    def __init__(self, *, provider: str, model: str, stop_reason: str) -> None:
        self.provider = provider
        self.model = model
        self.stop_reason = stop_reason
        self.summary = "stopped before it finished"
        super().__init__(
            f"{provider} stopped model {model!r} with stop_reason="
            f"{stop_reason!r}, which is not a natural end; the attempt failed."
        )


class GenerationCancelledError(IncompleteGenerationError):
    """The turn's generation was cut short by Stop."""

    summary = "was stopped"

    def __init__(self) -> None:
        super().__init__("The generation was stopped before it finished.")


def judge_generation(
    response: Any = None,
    *,
    error: Optional[BaseException] = None,
    cancelled: bool = False,
    provider: str = "",
    model: str = "",
) -> Optional[BaseException]:
    """Whether a generation may become an answer: ``None`` to accept it,
    otherwise the failed attempt it is.

    ``cancelled`` says Stop cut the generation short; ``error`` is what a call
    raised; otherwise ``response`` is what the call produced, judged by the
    stop reason its adapter attached. ``provider`` and ``model`` name the
    route in a failure's message.

    Only a natural-end stop reason accepts. A Stop, any error, a stop at an
    output limit, any other stop reason and no stop reason all fail. An
    error's verdict is the incomplete generation it carries, if any, else the
    error itself.
    """
    if cancelled:
        return GenerationCancelledError()
    if error is not None:
        return incomplete_generation(error) or error
    stop_reason = response_stop_reason(response)
    if stop_reason in NATURAL_STOP_REASONS:
        return None
    if stop_reason is None:
        return NoCompletionEvidenceError(provider=provider, model=model)
    if stop_reason in OUTPUT_LIMIT_STOP_REASONS:
        return OutputCapReachedError(
            provider=provider, model=model, cap=None, stop_reason=stop_reason,
        )
    return UnfinishedStopError(
        provider=provider, model=model, stop_reason=stop_reason,
    )


def failure_summary(failure: BaseException) -> str:
    """Why a failed attempt ended, short and content-free: the detail on the
    failed call's card in the chat. Never the failure's own text."""
    found = incomplete_generation(failure)
    return found.summary if found is not None else "failed"


def first_unfinished_attempt(
    route_errors: Iterable[BaseException],
) -> Optional[IncompleteGenerationError]:
    """The gate's verdict on a call whose routes all failed: the first route
    attempt that ended unfinished, if any.

    One is enough. The call then failed with an attempt that ended unfinished
    and no route's answer, whatever the other routes' errors were, and the
    turn is told the model did not finish. That holds whichever route came
    last, so the verdict is never the last route's error alone.
    """
    for error in route_errors:
        verdict = judge_generation(error=error)
        if isinstance(verdict, IncompleteGenerationError):
            return verdict
    return None


class UnfinishedAfterInlineToolsError(IncompleteGenerationError):
    """The gate rejected a response from a route that had already run tools
    inside its call.

    Such a route (the Codex app-server) runs a turn's tools through the
    ``tool_executor`` it is handed and reports them as
    ``executed_tool_calls``. They may have acted, so the route walk does not
    ask another route the same request, which could act again: the call fails
    instead of falling back. ``__cause__`` is the gate's verdict on the
    response. (A route that raises after its tools ran, rather than returning
    a response, is a separate defect: #3574.)
    """

    def __init__(self, provider: str, verdict: BaseException) -> None:
        self.provider = provider
        self.summary = failure_summary(verdict)
        super().__init__(
            f"Route {provider} ran tools inside its call and then did not "
            f"finish its response; not falling back to another route: {verdict}"
        )


def ran_inline_tools(response: Any) -> bool:
    """Whether ``response`` reports tools its route ran inside the call."""
    return bool(getattr(response, "executed_tool_calls", None))


# --------------------------------------------------------------------------
# The answer paths
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerPath:
    """One path by which a model's generation can become a turn's answer, or
    the completed-action checkpoint a turn records in its place.

    ``function`` is the code on the path, as ``module:qualname``. ``gates``
    are the calls it makes to reach this gate: :func:`judge_generation`
    itself, or the function of another registered path, by name. A gate
    ``name`` means exactly one call to ``name`` in the function, ``name*N``
    exactly ``N``: one per place the function's answer can come from, so a
    new place has to be registered too. A gate written ``name(keyword=True)``
    counts only calls that pass that keyword, and every call to ``name`` must.
    """

    name: str
    function: str
    gates: Tuple[str, ...]


ANSWER_PATHS: Tuple[AnswerPath, ...] = (
    # A route attempt: what one route's adapter returned, or the terminal
    # response that ended its stream.
    AnswerPath(
        "route_attempt",
        "kestrel_sovereign.llm.streaming:StreamingMixin._require_finished_generation",
        ("judge_generation",),
    ),
    AnswerPath(
        "route_attempt.streamed",
        "kestrel_sovereign.llm.streaming:StreamingMixin._stream_adapter_with_usage",
        ("_require_finished_generation",),
    ),
    # Fallback across routes: each route walk judges every attempt, so a later
    # route's answer stands alone and a failed route contributes nothing.
    AnswerPath(
        "route_walk",
        "kestrel_sovereign.llm.service:LLMService.generate_with_messages",
        ("_generate_with_messages_inner",),
    ),
    AnswerPath(
        "route_walk.attempts",
        "kestrel_sovereign.llm.service:LLMService._generate_with_messages_inner",
        # The managed remote GPU route, and each configured route.
        ("_require_finished_generation*2",),
    ),
    AnswerPath(
        "route_walk.streamed",
        "kestrel_sovereign.llm.streaming:StreamingMixin.stream_with_tool_detection",
        # Streamed: the managed remote GPU route, a route's tool stream and its
        # text-only stream. Not streamed: a route without a usage stream.
        (
            "_stream_adapter_with_usage(require_completion=True)*3",
            "_require_finished_generation",
        ),
    ),
    # Aggregate route results: the error a route walk raises once every route
    # failed states the gate's verdict on all of them.
    AnswerPath(
        "route_walk.aggregate",
        "kestrel_sovereign.llm.retry:state_aggregate_verdicts",
        ("first_unfinished_attempt",),
    ),
    AnswerPath(
        "route_walk.aggregate.verdict",
        "kestrel_sovereign.llm.generation_gate:first_unfinished_attempt",
        ("judge_generation",),
    ),
    # A turn's model call: bounded by its inactivity watchdog, and settled by
    # the gate when it raises (no answer; a completed tool batch's checkpoint).
    AnswerPath(
        "turn.model_call",
        "kestrel_sovereign.agent.orchestrator_engine:"
        "OrchestratorEngineMixin._generate_with_messages_bounded",
        ("generate_with_messages", "_settle_failed_generation"),
    ),
    AnswerPath(
        "turn.model_call.streamed",
        "kestrel_sovereign.agent.orchestrator_engine:"
        "OrchestratorEngineMixin._stream_with_tool_detection_bounded",
        ("stream_with_tool_detection", "_settle_failed_generation"),
    ),
    # Completed-action checkpoints: a call that failed after a tool batch
    # completed, and a turn stopped after one did.
    AnswerPath(
        "checkpoint.failed_call",
        "kestrel_sovereign.agent.orchestrator_engine:"
        "OrchestratorEngineMixin._settle_failed_generation",
        ("judge_generation",),
    ),
    AnswerPath(
        "checkpoint.stopped_turn",
        "kestrel_sovereign.agent.streaming:_stopped_generation",
        ("judge_generation",),
    ),
    # Non-streaming turns (``/api/agent/invoke``, signals, channels): the first
    # call, each tool-loop follow-up and the premature-yield repair.
    AnswerPath(
        "turn",
        "kestrel_sovereign.kestrel_agent:KestrelAgent.process_input",
        ("_process_input_traced_locked",),
    ),
    AnswerPath(
        "turn.first_call",
        "kestrel_sovereign.kestrel_agent:KestrelAgent._process_input_traced_locked",
        ("_generate_with_messages_bounded",),
    ),
    AnswerPath(
        "turn.follow_up",
        "kestrel_sovereign.agent.orchestrator_engine:"
        "OrchestratorEngineMixin._handle_orchestrator_response_impl",
        ("_generate_with_messages_bounded", "_repair_premature_turn_yield*2"),
    ),
    AnswerPath(
        "turn.repair",
        "kestrel_sovereign.agent.orchestrator_engine:"
        "OrchestratorEngineMixin._repair_premature_turn_yield",
        ("_generate_with_messages_bounded",),
    ),
    # Streamed turns (the chat UI).
    AnswerPath(
        "turn.streamed",
        "kestrel_sovereign.agent.streaming:StreamingMixin.process_input_streaming",
        ("_process_input_streaming_traced_locked",),
    ),
    AnswerPath(
        "turn.streamed.first_call",
        "kestrel_sovereign.agent.streaming:"
        "StreamingMixin._process_input_streaming_traced_locked",
        # Each point where a Stop is checked before the answer is recorded.
        ("_stream_with_tool_detection_bounded", "_stopped_generation*7"),
    ),
    AnswerPath(
        "turn.streamed.follow_up",
        "kestrel_sovereign.agent.orchestrator_engine:"
        "OrchestratorEngineMixin._handle_orchestrator_response_streaming",
        ("_stream_with_tool_detection_bounded", "_repair_premature_turn_yield*2"),
    ),
    # HTTP invoke: a turn whose model did not finish answers
    # ``502 generation_incomplete``, never a partial answer.
    AnswerPath(
        "http_invoke",
        "kestrel_sovereign.endpoints.agent:invoke_agent",
        ("process_input",),
    ),
    # A new agent's discovery conversation: its reply is the turn's answer
    # until discovery completes; a failed reply falls through to the turn.
    AnswerPath(
        "bootstrap.discovery",
        "kestrel_sovereign.bootstrap.service:BootstrapService.process_discovery_message",
        ("generate_with_messages",),
    ),
)


__all__ = [
    "ANSWER_PATHS",
    "AnswerPath",
    "GenerationCancelledError",
    "NATURAL_STOP_REASONS",
    "NoCompletionEvidenceError",
    "OUTPUT_LIMIT_STOP_REASONS",
    "UnfinishedAfterInlineToolsError",
    "UnfinishedStopError",
    "failure_summary",
    "first_unfinished_attempt",
    "judge_generation",
    "ran_inline_tools",
]
