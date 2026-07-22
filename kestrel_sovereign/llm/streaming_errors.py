"""One shared user-safe streaming-error formatting boundary (#2674 findings 3 & 4).

Every streaming transport that turns a mid-stream exception into client-visible
output MUST route through here, so the safe-text policy lives in exactly one
place and the ``/api/agent/stream`` (text/plain) and ``/api/bridge/stream`` (SSE)
paths cannot drift apart.

The security invariant: a streaming exception's text is UNTRUSTED. An adapter
that yields partial prose and THEN raises wraps that late exception, so its
``str(e)`` / ``underlying`` / ``provider`` can carry withheld response content
(under a strict fail-closed audit the whole turn is buffered) or an injected
marker. Terra reproduced both leaks:

* the bridge SSE ``error`` event emitted ``str(e)`` verbatim, leaking a
  ``BRIDGE_STRICT_WITHHELD_PROSE_MARKER`` (finding 3); and
* ``/api/agent/stream`` reflected ``LLMStreamingError.provider`` verbatim, and
  because that field accepts arbitrary strings at construction it leaked a
  ``ROUTE_FIELD_UNBOUNDED_MARKER__WITHHELD_TEXT`` (finding 4).

So the client message here is CONSTANT per error class — it never interpolates
the exception's message, underlying, or provider. The failing route and full
error stay operator-log only (a separate trust boundary the callers own).

Callers:
    * :func:`safe_streaming_error_message` — the core content-free string.
    * :func:`agent_stream_error_block` — the ``/api/agent/stream`` text/plain
      markdown envelope.
    * :func:`bridge_sse_error_event` — the ``/api/bridge/stream`` SSE ``data:``
      event (JSON ``{"type":"error","message": ...}``).
"""

import json


# Each safe message is a ``(header, body)`` pair: the ``header`` is the bold
# summary the chat error card renders, the ``body`` the plain guidance after it.
# Both are CONSTANT per error class — no interpolation of ``str(exc)``,
# ``underlying``, or ``provider`` ever.

# Any non-``LLMStreamingError`` exception is arbitrary, untrusted text (a
# mid-buffer failure under a strict audit could carry the withheld response), so
# the client only ever gets this constant.
_GENERIC_ERROR = (
    "Error generating response.",
    "The request could not be completed. Please try again; if it persists, "
    "check the server logs.",
)

# The model-route failure message. We deliberately do NOT name the failing route:
# ``LLMStreamingError.provider`` is an unvalidated free string accepted at
# construction (#2674 finding 4 — it leaked ROUTE_FIELD_UNBOUNDED_MARKER…). A
# CONSTANT "your selected model route" label preserves the actionable
# no-blind-fallback / recovery guidance ([[feedback_no_blind_fallbacks]]) without
# reflecting any attacker- or content-controlled text. The failing route stays in
# the operator log (a separate trust boundary the caller owns).
_ROUTE_ERROR = (
    "Your selected model route failed.",
    "No fallback response was generated — you selected this route explicitly. "
    "To recover, pick a different model/route from the dropdown, or fix the "
    "underlying issue (auth token, quota, etc.). The specific error is in the "
    "server logs.",
)


def _is_llm_streaming_error(exc: BaseException) -> bool:
    """True when ``exc`` is an :class:`LLMStreamingError` (a selected-route
    failure), imported lazily so this module stays free of heavy LLM imports.
    Falls back to ``False`` if the class can't be imported for any reason —
    the safest classification (generic constant message)."""
    try:
        from kestrel_sovereign.llm.streaming import LLMStreamingError
    except Exception:  # pragma: no cover - defensive import guard
        return False
    return isinstance(exc, LLMStreamingError)


def _classify(exc: BaseException):
    """Map ``exc`` to its ``(header, body)`` safe message pair."""
    return _ROUTE_ERROR if _is_llm_streaming_error(exc) else _GENERIC_ERROR


def safe_streaming_error_message(exc: BaseException) -> str:
    """The single content-free, user-safe message for a failed stream.

    CONSTANT per error class — never interpolates ``str(exc)``, ``underlying``,
    or ``provider``. A selected-route failure returns the recovery guidance; any
    other exception returns the generic message. Full detail is the caller's job
    to log operator-side.
    """
    header, body = _classify(exc)
    return f"{header} {body}"


def agent_stream_error_block(exc: BaseException) -> str:
    """The ``/api/agent/stream`` text/plain markdown envelope for a stream error.

    Renders the ``(header, body)`` pair as the chat client's error card
    (``⚠️ **header** body``). The message content is single-sourced with the
    bridge path so the two cannot diverge on what is (or isn't) safe to reveal.
    """
    header, body = _classify(exc)
    return f"\n\n---\n⚠️ **{header}** {body}"


def bridge_sse_error_event(exc: BaseException) -> str:
    """The ``/api/bridge/stream`` SSE ``data:`` event for a stream error.

    A stable, safe payload — ``{"type":"error","message": <constant>}`` — never
    the underlying/message/provider text. Same safe message as the agent path.
    """
    payload = json.dumps({
        "type": "error",
        "message": safe_streaming_error_message(exc),
    })
    return f"data: {payload}\n\n"
