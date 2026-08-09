"""Resolve the chat session that deferred work was registered from (#2877).

Anything that finishes *later* — a restart request, a watched wait handle, a
dispatched Talon job — has to carry the session it originated from, or its
completion wake has no thread to land in. The dispatcher routes a signal's
``session_id`` into ``process_input``; when that is absent, the conversation
store falls back to its 30-minute implicit-session heuristic
(``AsyncConversationStore._derive_implicit_session_id``) and mints a FRESH
session id whenever the previous message is older than that window. That is
what stranded hours of autonomous Talon work in wake-only sessions nobody was
watching, and why the failure looked intermittent: a wake that happened to
land within 30 minutes of the last message inherited the right session by
accident, while a slower job did not.

One resolver, used by every registration point, so the three call sites cannot
drift apart in what "the originating session" means.
"""

from __future__ import annotations

from typing import Any


def resolve_origin_session_id(agent: Any) -> str:
    """Return the session id the current turn is running under, or ``""``.

    Prefers the agent's authoritative per-turn ``_active_session_id`` (set by
    both the streaming and non-streaming turn bodies from the effective
    session, including the JSON-body session the primary chat path uses) and
    falls back to the logging ``session_id_var`` ContextVar, which is set only
    from a query param / header.

    An empty string means genuinely session-less — a CLI or system-initiated
    caller with no observer thread. Callers must treat that as "no binding"
    rather than substituting a session of their own invention.

    Only a genuine non-empty ``str`` counts. Anything else (a test double's
    auto-created attribute, a stray sentinel) is NOT a session id, and
    coercing it would bind a wake to a thread that does not exist.
    """
    if agent is None:
        return ""
    active = getattr(agent, "_active_session_id", None)
    if isinstance(active, str) and active:
        return active
    try:
        from kestrel_sovereign.logging_config import session_id_var

        ambient = session_id_var.get()
    except Exception:
        return ""
    return ambient if isinstance(ambient, str) and ambient else ""
