"""Request-scoped identity for LLM invocation telemetry.

The service used to keep these values in one mutable dictionary on the
``LLMService`` instance.  A service is shared by every concurrent request for
an agent, so whichever request wrote that dictionary last supplied the
session/user identity for every in-flight completion.  This module keeps the
compatibility setter task-local and gives generation callers an immutable,
explicit context they can pass through the invocation boundary.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

from kestrel_sovereign.logging_config import correlation_id_var, session_id_var


@dataclass(frozen=True, slots=True)
class LLMInvocationContext:
    """Correlation identity captured once for one logical LLM invocation."""

    session_id: Optional[str] = None
    companion_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None


_AMBIENT_INVOCATION_CONTEXT: ContextVar[LLMInvocationContext] = ContextVar(
    "kestrel_llm_invocation_context",
    default=LLMInvocationContext(),
)


def set_ambient_invocation_context(context: LLMInvocationContext) -> None:
    """Set the compatibility context for the current async task.

    New call sites should pass ``LLMInvocationContext`` explicitly.  The
    ambient lane exists for the historical ``set_observability_context`` API;
    ``ContextVar`` inheritance keeps it compatible without sharing mutable
    request state between concurrent tasks.
    """

    _AMBIENT_INVOCATION_CONTEXT.set(context)


def _first_defined(*values: Optional[str]) -> Optional[str]:
    """Return the first value supplied, preserving explicit empty strings."""

    return next((value for value in values if value is not None), None)


def resolve_invocation_context(
    context: Optional[LLMInvocationContext] = None,
    *,
    session_id: Optional[str] = None,
) -> LLMInvocationContext:
    """Resolve one immutable context from explicit, ambient, and HTTP state.

    Explicit fields win.  The existing ``session_id`` generation parameter is
    authoritative when present, and request middleware context only fills
    values the caller did not supply.
    """

    explicit = context if context is not None else LLMInvocationContext()
    ambient = _AMBIENT_INVOCATION_CONTEXT.get()
    return LLMInvocationContext(
        session_id=_first_defined(
            session_id,
            explicit.session_id,
            ambient.session_id,
            session_id_var.get(),
        ),
        companion_id=_first_defined(explicit.companion_id, ambient.companion_id),
        user_id=_first_defined(explicit.user_id, ambient.user_id),
        correlation_id=_first_defined(
            explicit.correlation_id,
            ambient.correlation_id,
            correlation_id_var.get(),
        ),
    )
