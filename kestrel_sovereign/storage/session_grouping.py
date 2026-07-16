"""Shared session-boundary algorithm (#2019).

Both the user-facing ``GET /api/conversations`` endpoint and the agent-facing
``list_conversations`` memory tool need to turn a flat, ordered list of
conversation messages into *sessions*. If they computed those boundaries
independently they would drift, and the agent could soft-delete a "session"
that the UI groups differently. This module is the single source of truth so
the two surfaces always agree on where one conversation ends and the next
begins.

The function is pure (no I/O, no decryption, no privacy concerns): each caller
fetches rows however it must — the endpoint through the privacy-wrapped
storage, the store method through the conversation store — normalizes them into
plain dicts, and hands them here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from kestrel_sovereign.kestrel_config.constants import SESSION_GAP_MINUTES


def coerce_session_timestamp(created_at: Any) -> Optional[datetime]:
    """Parse a stored timestamp into one comparable naive-UTC datetime.

    Mirrors the lenient parsing the conversations endpoint has always used:
    accept datetimes as-is, try the historical SQL/ISO string formats, and
    fall back to :meth:`datetime.fromisoformat` for ``Z``/offset-bearing values.

    SQLite conversation history legitimately contains a mixture of naive SQL
    timestamps and ISO-8601 values.  Treat naive values as UTC and normalize
    aware values to naive UTC so sorting and gap arithmetic can never raise on
    an aware/naive mixture.  ``None`` means chronology cannot be established;
    presentation callers may substitute a clock, while destructive callers
    must fail closed.
    """
    parsed: Optional[datetime]
    if isinstance(created_at, datetime):
        parsed = created_at
    elif isinstance(created_at, str):
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                parsed = datetime.strptime(created_at, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            try:
                parsed = datetime.fromisoformat(
                    created_at[:-1] + "+00:00"
                    if created_at.endswith("Z")
                    else created_at
                )
            except ValueError:
                return None
    else:
        return None

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def group_messages_into_sessions(
    messages: Iterable[Dict[str, Any]],
    gap_minutes: float = SESSION_GAP_MINUTES,
    now: Optional[datetime] = None,
    keep_empty_markers: bool = False,
    collect_messages: bool = False,
) -> List[Dict[str, Any]]:
    """Group ordered messages into session clusters.

    Args:
        messages: iterable of message dicts ordered **oldest-first**, each with:
            - ``id``: message row id (used as the session_id fallback for
              legacy time-gap clusters that carry no metadata session_id).
            - ``role``: ``'user'`` / ``'assistant'`` / ...
            - ``content``: message text (used only to populate the preview).
            - ``metadata``: dict, possibly carrying ``new_session`` and/or a
              canonical ``session_id``.
            - ``created_at``: datetime or ISO/SQL string.
        gap_minutes: minutes of inactivity that start a new session.
        now: clock used when a row has an unparseable/missing timestamp;
            defaults to ``datetime.now()`` (injectable for tests).
        keep_empty_markers: when ``True``, a session established solely by a
            ``new_session`` marker row (no real messages yet) is still returned,
            with ``message_count == 0`` (#2222). A freshly-created conversation
            is a real, list-visible session the moment the user starts it — the
            UI prepends a tile for it and the reconciling refetch must find it
            server-side, or the tile vanishes. The agent-facing memory tools
            leave this ``False`` so empty just-started sessions stay out of
            recall.
        collect_messages: when ``True``, each returned session additionally
            carries a ``messages`` list holding the (normalized) message dicts
            attributed to it — structural ``new_session`` marker rows excluded.
            Content search needs the message→session attribution this boundary
            algorithm computes; exposing it here keeps search on the same
            single source of truth instead of re-deriving boundaries.

    Returns:
        list of session dicts ordered **oldest-first**, each with:
            ``session_id``, ``started_at`` (iso), ``last_message_at`` (iso),
            ``message_count``, ``user_message_count``,
            ``preview_content`` (raw, undecorated), ``preview_metadata`` (dict),
            and — only when ``collect_messages`` — ``messages``.
        Callers reverse / slice / decorate as needed.
    """
    sessions: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    # Whether ``current`` was established by a ``new_session`` marker row and has
    # accumulated no real messages yet — the retain-if-``keep_empty_markers``
    # case (#2222).
    current_is_empty_marker = False

    def _keep(session: Dict[str, Any], is_empty_marker: bool) -> bool:
        return session["message_count"] > 0 or (keep_empty_markers and is_empty_marker)

    def _new_session(msg_id: Any, ts: datetime, session_uuid: Optional[str]) -> Dict[str, Any]:
        # Canonical identity (#2012): prefer the session's own metadata
        # session_id (a UUID minted by the store) so the value surfaces match
        # where messages are actually filed. Fall back to the first message's
        # row id only for genuinely legacy clusters with no metadata session_id.
        session: Dict[str, Any] = {
            "session_id": str(session_uuid) if session_uuid else str(msg_id),
            "started_at": ts.isoformat(),
            "last_message_at": ts.isoformat(),
            "message_count": 0,
            "user_message_count": 0,
            "preview_content": None,
            "preview_metadata": None,
        }
        if collect_messages:
            session["messages"] = []
        return session

    for msg in messages:
        msg_id = msg.get("id")
        role = msg.get("role")
        content = msg.get("content")
        meta = msg.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}

        timestamp = (
            coerce_session_timestamp(msg.get("created_at"))
            or coerce_session_timestamp(now)
            or datetime.now(timezone.utc).replace(tzinfo=None)
        )

        is_new_session_marker = bool(meta.get("new_session"))
        meta_session_id = None
        sid = meta.get("session_id")
        # Only treat non-integer values as a canonical UUID. A bare-integer
        # session_id is a mis-filed legacy key (#2012) where the row-id fallback
        # groups more stably.
        if sid and not str(sid).isdigit():
            meta_session_id = sid

        if current is None:
            current = _new_session(msg_id, timestamp, meta_session_id)
            current_is_empty_marker = False

        last_ts = datetime.fromisoformat(current["last_message_at"])
        gap = (timestamp - last_ts).total_seconds() / 60

        # A change of canonical session_id starts a new session even within the
        # gap window and with no explicit marker — otherwise a summary would be
        # labeled with one id while its counts/preview include the next session,
        # so lifecycle tools would act on a different scope than the list shows
        # (#2019). None ids (unlabeled turns) stay with the current session.
        #
        # Only split when the CURRENT session already carries a real id (a UUID,
        # not a legacy row-id fallback). A legacy cluster (numeric anchor) is
        # resolved by ``_get_session_messages`` via a forward time-walk that
        # does NOT stop on id changes, so it absorbs a following UUID row; if we
        # split the list there, deleting the listed legacy session would also
        # destroy the UUID session the list showed as separate. Keeping them
        # merged matches what the delete resolver actually touches. Two distinct
        # UUID sessions DO split — there the resolver matches by metadata
        # membership, so each delete stays scoped to its own id.
        session_changed = (
            meta_session_id is not None
            and current["session_id"] != meta_session_id
            and not str(current["session_id"]).isdigit()
        )

        if gap > gap_minutes or is_new_session_marker or session_changed:
            if _keep(current, current_is_empty_marker):
                sessions.append(current)
            current = _new_session(msg_id, timestamp, meta_session_id)
            # The explicit marker row is structural, not a real message. Track
            # that this new session so far exists ONLY because of it, so
            # ``keep_empty_markers`` can retain a just-started conversation with
            # no messages yet (#2222).
            current_is_empty_marker = is_new_session_marker
            if is_new_session_marker:
                continue

        current["message_count"] += 1
        current["last_message_at"] = timestamp.isoformat()
        if collect_messages:
            current["messages"].append(msg)
        if role == "user":
            current["user_message_count"] += 1
            # Operator-signal fallback notices (#operator_signals.py) are
            # persisted with role="user" so they replay in history, but they
            # are synthetic system chatter, not something the user typed —
            # skip them when picking the preview so an auto-mode/budget/
            # governance notice never becomes the conversation's title.
            if current["preview_content"] is None and not meta.get("operator_signal"):
                current["preview_content"] = content
                current["preview_metadata"] = meta

    if current and _keep(current, current_is_empty_marker):
        sessions.append(current)

    return sessions


def coalesce_sessions_by_session_id(
    sessions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge summaries that share the same canonical ``session_id`` (#2019).

    :func:`group_messages_into_sessions` splits a conversation on every
    ``> gap_minutes`` lull, so a session that was *resumed* past that gap (the
    same UUID re-supplied as ``session_id``) produces several clusters carrying
    an identical ``session_id``. That is harmless for a purely visual timeline,
    but both the agent lifecycle tools and the UI delete/restore/purge **by**
    ``session_id`` — the resolver matches the whole UUID and touches every one
    of those rows. If the list surfaced two entries with the same id, deleting
    "one" would silently destroy the other. Coalescing guarantees each returned
    ``session_id`` is a unique, faithful delete target.

    Input and output are oldest-first. Legacy clusters that fell back to a
    row-id are inherently unique, so they pass through untouched.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for session in sessions:
        sid = session["session_id"]
        existing = merged.get(sid)
        if existing is None:
            merged[sid] = dict(session)
            order.append(sid)
            continue
        existing["message_count"] += session["message_count"]
        existing["user_message_count"] += session["user_message_count"]
        if "messages" in existing and "messages" in session:
            existing["messages"] = existing["messages"] + session["messages"]
        if session["started_at"] < existing["started_at"]:
            existing["started_at"] = session["started_at"]
        if session["last_message_at"] > existing["last_message_at"]:
            existing["last_message_at"] = session["last_message_at"]
        # The first occurrence is the oldest cluster, so its preview (the
        # earliest user message) is already retained — unless that cluster's
        # only user row was a skipped operator-signal notice, leaving its
        # preview None. In that case fall through to a later cluster's real
        # preview so a resumed session never shows an empty title.
        if existing["preview_content"] is None and session["preview_content"] is not None:
            existing["preview_content"] = session["preview_content"]
            existing["preview_metadata"] = session["preview_metadata"]
    return [merged[sid] for sid in order]


def summarize_sessions(
    messages: Iterable[Dict[str, Any]],
    names: Optional[Dict[str, str]] = None,
    limit: int = 50,
    include_trashed: bool = False,
    preview_chars: int = 80,
    preview_transform: Optional[Callable[[str], str]] = None,
) -> List[Dict[str, Any]]:
    """Turn ordered messages into navigable, newest-first session summaries.

    The single shaping path behind the agent's ``list_conversations`` tool —
    used by both the conversation store (live rows) and the privacy wrapper
    (in-memory ISOLATED rows) so the summary shape never diverges. Groups,
    coalesces same-UUID clusters into unique delete targets (#2019), trims to
    ``limit``, and replaces the raw preview fields with a short plaintext
    ``preview`` plus an ``is_trashed`` flag and an optional ``name``.

    Args:
        messages: message dicts ordered oldest-first (see
            :func:`group_messages_into_sessions`).
        names: optional ``{session_id: title}`` map to decorate summaries.
        limit: maximum number of sessions (newest-first).
        include_trashed: value stamped onto each summary's ``is_trashed``.
        preview_chars: preview truncation length.
    """
    names = names or {}
    grouped = coalesce_sessions_by_session_id(group_messages_into_sessions(messages))
    # Newest-first by last activity. Sorting on last_message_at (not list
    # position) so a conversation resumed past the gap ranks by its latest
    # message, never buried under older threads or dropped by ``limit`` (#2019).
    grouped.sort(key=lambda s: s["last_message_at"], reverse=True)
    sessions = grouped[:limit]
    for session in sessions:
        preview = session.pop("preview_content", None) or ""
        session.pop("preview_metadata", None)
        if preview and preview_transform is not None:
            # e.g. unwrap sent-form so the preview is the raw user text.
            preview = preview_transform(preview)
        if preview:
            preview = preview[:preview_chars] + ("..." if len(preview) > preview_chars else "")
        session["preview"] = preview
        session["is_trashed"] = include_trashed
        sid = session.get("session_id")
        if sid in names:
            session["name"] = names[sid]
    return sessions
