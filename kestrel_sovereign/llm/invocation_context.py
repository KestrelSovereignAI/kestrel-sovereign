"""Request-scoped identity for LLM invocation telemetry.

The service used to keep these values in one mutable dictionary on the
``LLMService`` instance.  A service is shared by every concurrent request for
an agent, so whichever request wrote that dictionary last supplied the
session/user identity for every in-flight completion.  This module keeps the
compatibility setter task-local and gives generation callers an immutable,
explicit context they can pass through the invocation boundary.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator, Optional

from kestrel_sovereign.logging_config import correlation_id_var, session_id_var


@dataclass(frozen=True, slots=True)
class LLMInvocationContext:
    """Correlation identity captured once for one logical LLM invocation."""

    session_id: Optional[str] = None
    companion_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None


class LLMInvocationContextState:
    """Service-local ambient context with explicit lifetime management.

    A module-global ``ContextVar`` isolates concurrent tasks, but it does not
    isolate two ``LLMService`` instances used by the *same* task.  Kestrel can
    host several agents in one process, so each service owns one of these
    state objects.  ``scope`` is the preferred compatibility lane: child tasks
    inherit the request identity, while the parent is restored when the scope
    exits.
    """

    def __init__(self) -> None:
        self._ambient: ContextVar[LLMInvocationContext] = ContextVar(
            f"kestrel_llm_invocation_context_{id(self):x}",
            default=LLMInvocationContext(),
        )

    def get(self) -> LLMInvocationContext:
        """Return the current task's ambient identity."""

        return self._ambient.get()

    def set(self, context: LLMInvocationContext) -> Token[LLMInvocationContext]:
        """Install ``context`` and return a token that can restore its parent."""

        return self._ambient.set(context)

    def reset(self) -> None:
        """Clear sticky compatibility state for the current task."""

        self._ambient.set(LLMInvocationContext())

    @contextmanager
    def scope(
        self, context: LLMInvocationContext
    ) -> Iterator[LLMInvocationContext]:
        """Install ``context`` only for the dynamic extent of this scope."""

        token = self.set(context)
        try:
            yield context
        finally:
            self._ambient.reset(token)


# Preserve the short-lived module API introduced with #2510 for callers that
# use this helper directly.  LLMService deliberately never touches this state:
# every service supplies its own ``ambient=...`` snapshot below.
_COMPATIBILITY_INVOCATION_CONTEXT_STATE = LLMInvocationContextState()


def set_ambient_invocation_context(context: LLMInvocationContext) -> None:
    """Set the standalone compatibility context for the current async task."""

    _COMPATIBILITY_INVOCATION_CONTEXT_STATE.set(context)


def _first_defined(*values: Optional[str]) -> Optional[str]:
    """Return the first value supplied, preserving explicit empty strings."""

    return next((value for value in values if value is not None), None)


def resolve_invocation_context(
    context: Optional[LLMInvocationContext] = None,
    *,
    ambient: Optional[LLMInvocationContext] = None,
    session_id: Optional[str] = None,
) -> LLMInvocationContext:
    """Resolve one immutable context from explicit, ambient, and HTTP state.

    Explicit fields win.  The existing ``session_id`` generation parameter is
    authoritative when present, and request middleware context only fills
    values the caller did not supply.
    """

    explicit = context if context is not None else LLMInvocationContext()
    ambient = (
        ambient
        if ambient is not None
        else _COMPATIBILITY_INVOCATION_CONTEXT_STATE.get()
    )
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
