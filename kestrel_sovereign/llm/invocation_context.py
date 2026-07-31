"""Request-scoped identity for LLM invocation telemetry.

The service used to keep these values in one mutable dictionary on the
``LLMService`` instance.  A service is shared by every concurrent request for
an agent, so whichever request wrote that dictionary last supplied the
session/user identity for every in-flight completion.  This module keeps the
compatibility setter task-local and gives generation callers an immutable,
explicit context they can pass through the invocation boundary.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator, Optional

from kestrel_sovereign.logging_config import correlation_id_var, session_id_var


@dataclass(frozen=True, slots=True)
class LLMInvocationContext:
    """Correlation identity captured once for one logical LLM invocation.

    ``redact_content`` (#2674 finding 3) is a per-invocation, immutable
    content-redaction flag threaded through this same boundary. When an
    *enforcing* (strict / fail-closed) POST_RESPONSE audit governs the turn,
    the assistant prose is withheld from the user until the audit verdict —
    but the raw provider prompt/response would otherwise land in durable
    ``llm_calls`` telemetry (``user_prompt`` / ``response_preview``) and the
    OpenTelemetry LLM span (``input.value`` / ``output.value``) BEFORE that
    verdict, surviving even a DENY. The audit's own provider call is the same
    hazard: its ``user_prompt`` IS the withheld assistant prose. Setting this
    on the frozen context for those calls makes the telemetry sinks blank the
    prompt/response content while preserving every content-free field (usage,
    timing, provider, model, error classification). It rides the frozen
    context — never global/ambient state — so a concurrent or background LLM
    call on the same service is unaffected.
    """

    session_id: Optional[str] = None
    companion_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    redact_content: bool = False


_EMPTY_INVOCATION_CONTEXT = LLMInvocationContext()


class LLMInvocationContextState:
    """Service-local ambient context with explicit lifetime management.

    A module-global ``ContextVar`` isolates concurrent tasks, but it does not
    isolate two ``LLMService`` instances used by the *same* task.  Kestrel can
    host several agents in one process, so each service owns one of these
    state objects.  ``scope`` is the preferred compatibility lane: child tasks
    inherit the request identity, while the parent is restored when the scope
    exits.

    The ``ContextVar`` alone is *task-local*: an identity written by
    :meth:`set` reaches the current task and any child task spawned *after*
    the write, but never a task spawned *before* it, a sibling task tree, or a
    thread offload.  #2569 — a downstream non-stream chat handler set identity
    in its request task and then dispatched the LLM call through a task that
    never inherited it, so the metering callback silently no-oped with
    ``companion_id``/``user_id`` both ``None``.

    The obvious "readable from anywhere" repair — a single last-writer
    snapshot on the instance — is exactly the cross-request bleed #2510 was
    filed to kill: two concurrent requests share one service, so B's ``set``
    would silently re-bill A's tokens to B.  We refuse that.  Instead the
    cross-task fallback is keyed on ``session_id``, the one discriminator that
    *does* survive the boundary — it threads through as an explicit generation
    argument (``generate_with_messages(session_id=...)``), not a ContextVar.
    :meth:`set` records ``session_id -> context`` in a bounded LRU, and
    :meth:`get_for_session` recovers the identity for *that* session only.  Two
    concurrent requests carry distinct session ids, so neither can read the
    other's identity, and a call that never set a context (background audit,
    reflection) resolves nothing for its session and correctly no-ops.  The
    registry is per-instance, so two services in one task never share identity;
    the LRU bound ages out finished sessions without any caller-side cleanup.
    """

    #: Cap on remembered ``session_id -> context`` entries.  Bounds memory and
    #: ages out finished sessions; a *new* session never matches an old key, so
    #: eviction can only lose recovery for a session that already went quiet.
    _MAX_TRACKED_SESSIONS = 1024

    def __init__(self) -> None:
        self._ambient: ContextVar[LLMInvocationContext] = ContextVar(
            f"kestrel_llm_invocation_context_{id(self):x}",
            default=_EMPTY_INVOCATION_CONTEXT,
        )
        self._by_session: "OrderedDict[str, LLMInvocationContext]" = OrderedDict()

    def get(self) -> LLMInvocationContext:
        """Return the current task's ambient identity (task-local only)."""

        return self._ambient.get()

    def get_for_session(
        self, session_id: Optional[str]
    ) -> Optional[LLMInvocationContext]:
        """Recover the identity a prior :meth:`set` recorded for ``session_id``.

        This is the cross-task compatibility lane: when the dispatching task
        never inherited the setter's ContextVar, the resolver falls back to the
        identity keyed on the session id that *did* thread through explicitly.
        Returns ``None`` when nothing was recorded for the session, so a call
        with no matching session identity no-ops instead of borrowing another
        request's billing identity.
        """

        if not session_id:
            return None
        context = self._by_session.get(session_id)
        if context is not None:
            # Refresh recency so an active conversation isn't evicted mid-flight.
            self._by_session.move_to_end(session_id)
        return context

    def set(self, context: LLMInvocationContext) -> Token[LLMInvocationContext]:
        """Install ``context`` and return a token that can restore its parent.

        When ``context`` carries a ``session_id``, also record it in the
        per-session registry so a request's identity survives a downstream task
        boundary *without* a process-wide last-writer snapshot.
        """

        if context.session_id:
            self._remember_session(context)
        return self._ambient.set(context)

    def _remember_session(self, context: LLMInvocationContext) -> None:
        session_id = context.session_id
        assert session_id is not None  # guarded by caller
        self._by_session[session_id] = context
        self._by_session.move_to_end(session_id)
        while len(self._by_session) > self._MAX_TRACKED_SESSIONS:
            self._by_session.popitem(last=False)

    def reset(self) -> None:
        """Clear the current task's ambient identity.

        Task-local only: it resets *this* task's ContextVar and never touches
        the shared per-session registry, so one request ending cannot wipe a
        concurrent request's cross-task fallback identity.
        """

        self._ambient.set(_EMPTY_INVOCATION_CONTEXT)

    @contextmanager
    def scope(
        self, context: LLMInvocationContext
    ) -> Iterator[LLMInvocationContext]:
        """Install ``context`` only for the dynamic extent of this scope.

        A scope is a bounded, task-local override: the ContextVar is restored
        to its parent on exit and the per-session registry is left untouched,
        so no recovery state can outlive the scope.
        """

        token = self._ambient.set(context)
        try:
            yield context
        finally:
            self._ambient.reset(token)


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
    # LLMService always supplies its own ``ambient=...`` snapshot; a caller that
    # omits one contributes no ambient identity of its own.
    ambient = ambient if ambient is not None else _EMPTY_INVOCATION_CONTEXT
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
        # #2674 finding 3: redaction is a fail-safe flag — OR it across the
        # merged sources so a resolve step can never silently drop a caller's
        # request to withhold prompt/response content from telemetry.
        redact_content=bool(explicit.redact_content or ambient.redact_content),
    )
