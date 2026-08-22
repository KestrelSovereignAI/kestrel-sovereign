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

import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from kestrel_sovereign.kestrel_config.constants import SESSION_GAP_MINUTES


def signal_wake_source(metadata: Dict[str, Any]) -> Optional[str]:
    """Return the wake source when a row is an autonomous signal wake (#2204).

    COGNITION signal wakes (heartbeat, ``talon.job_complete``,
    ``wait.complete``, ``restart.completed``, a2a) are persisted with
    ``role="user"`` so they replay in history, and tagged
    ``metadata.signal_wake = {"source", "mode"}`` by the dispatcher. Returns
    ``None`` for ordinary rows. A wake whose source is missing falls back to
    the same generic ``"signal"`` label the transcript chip uses, so the row
    stays recognizable as autonomous rather than silently anonymous.
    """
    wake = metadata.get("signal_wake")
    if not wake:
        return None
    if isinstance(wake, dict):
        source = wake.get("source")
        if source:
            return str(source)
    return "signal"


def autonomous_wake_preview(source: Optional[str]) -> str:
    """Title a session whose only user rows were autonomous signal wakes.

    Such a session has no human turn to preview, but it is not a blank "New
    conversation" either — autonomous work actually ran in it. Name it for
    what it is, the same honesty the transcript's "Autonomous wake" chip
    already applies to the individual rows (#2947).
    """
    return f"Autonomous wake — {source or 'signal'}"


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
        # The gate goes BEFORE every attempt, not just the last one. Both
        # parsers below read strings the ordering key cannot: `strptime`
        # compiles its format with `re.IGNORECASE`, so it takes a lowercase
        # `t` separator that `julianday` rejects, and `fromisoformat` takes
        # more still. Guarding only the fallback left that one through —
        # measured, and the reason the first version of this fix was incomplete.
        if not _JULIANDAY_READABLE.match(created_at):
            return None
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                parsed = datetime.strptime(created_at, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            # Guarded by what the ORDERING can read, because `fromisoformat`
            # accepts a good deal more than `julianday` does — the basic form
            # (`20260101T110000`), a `+0500` offset, a lowercase `t` —
            # incidental permissiveness, not a format this codebase writes or
            # this function documents. Accepting it made the parser's domain
            # wider than the SQL ordering key's: `julianday` returns NULL for
            # it, so such a row sorted at the far end of the canonical order and
            # `LIMIT` could drop it from the conversation list entirely, while
            # every Python path treated it as a perfectly good 2026 timestamp
            # (round-18 review).
            #
            # Narrowed rather than normalized in SQL: one domain defined by what
            # the ordering can express beats a normalization kept in step with a
            # parser, which is how the two came apart in the first place. A
            # value outside it is undatable in BOTH, and an undatable row has a
            # defined home — the stamp of the row before it.
            try:
                parsed = datetime.fromisoformat(
                    created_at[:-1] + "+00:00"
                    if created_at.endswith("Z")
                    else created_at
                )
            except ValueError:
                return None
        if parsed is None:
            return None
    else:
        return None

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def timestamp_query_param(backend_type: str, value: Any) -> Any:
    """Bind a timestamp in the spelling the column it is compared against uses.

    Binding an aware ``datetime`` on SQLite would otherwise go through Python's
    deprecated implicit adapter, and PostgreSQL wants a naive UTC ``datetime``
    for asyncpg. Those two were always the job; the spelling is the part that
    changed with #3009.

    It used to be ``value.isoformat()`` on SQLite — a ``T`` separator — which
    was safe only because every comparison ran through ``julianday`` on both
    sides. ``created_at`` now holds exactly one spelling, and that is what
    makes a raw text comparison correct; a parameter in a different spelling
    would be the one value in the comparison that still needed converting. So
    it is rendered by the same module the column's CHECK is computed from.

    Sub-second precision is KEPT, which is why this is ``comparable_`` and not
    the storage spelling. Nothing stores a fraction, but a boundary may carry
    one, and rounding it down moves the boundary — ``purge_all_since`` compares
    ``>=`` and would permanently delete a row that predates the watermark it
    was given.

    A value nothing can date is passed through unchanged rather than becoming
    NULL: a predicate against it found nothing before and must go on finding
    nothing, rather than quietly matching every row with a NULL comparison.
    """
    if backend_type == "postgres":
        parsed = coerce_session_timestamp(value)
        return value if parsed is None else parsed
    if backend_type == "sqlite":
        from .conversation_created_at import comparable_created_at

        return comparable_created_at(value) or value
    return value


def canonical_timestamp_sql(backend_type: str, expression: str) -> str:
    """Normalize a timestamp SQL expression for the active backend."""
    if backend_type == "sqlite":
        return f"julianday({expression})"
    return expression


def bytewise_sql(backend_type: str, expression: str) -> str:
    """Compare a text expression by code point, as Python's ``sorted`` does.

    PostgreSQL databases are commonly created with a locale-aware default
    collation — ``en_US.utf8`` here — under which punctuation and case are not
    primary distinctions. Measured: ``('A-1','a-1','A_1','ab1')`` sorts
    ``a-1,A-1,A_1,ab1`` under that collation and ``A-1,A_1,a-1,ab1`` under
    ``"C"``, which is what Python produces. Session ids may contain uppercase
    letters, hyphens and underscores, so an ordering meant to be shared between
    a SQL page and a Python sort has to name the comparison rather than inherit
    whichever one the cluster was initialised with.

    SQLite's default is already bytewise, and it has no ``COLLATE "C"``, so the
    expression is returned unchanged there.
    """
    if backend_type == "postgres":
        return f'{expression} COLLATE "C"'
    return expression


def timestamp_predicate(backend_type: str, column: str, operator: str) -> str:
    """Compare timestamp columns and parameters across supported backends."""
    if operator not in {"<", ">", ">="}:
        raise ValueError(f"Unsupported timestamp comparison: {operator}")
    left = canonical_timestamp_sql(backend_type, column)
    right = canonical_timestamp_sql(backend_type, "?")
    return f"{left} {operator} {right}"


#: Substituted for a stamp when a transcript carries no readable one at all.
#: Any instant would do; what matters is that it is the SAME instant every time,
#: so two groupings of one transcript agree.
_GROUPING_EPOCH = datetime(1970, 1, 1)

#: The same constant, for the OTHER implementation of session membership.
#:
#: ``AsyncConversationStore._filter_session_rows`` re-derives the gap rules for
#: reads and lifecycle snapshots, and it used to date an unreadable timestamp
#: with ``datetime.now()`` — as this function did, which is why the two agreed.
#: They stopped agreeing when this one became a function of the rows, which the
#: #2959 projection requires: a grouping that consults a clock cannot be cached,
#: because re-deriving it later gives a different answer.
#:
#: Sharing the constant does not make the two agree about WHICH session an
#: unreadable row joins — this one inherits its predecessor and the other cannot
#: always see one. What it does remove is the part that made the answer depend
#: on when you asked. The disagreement that remains is one mechanism with two
#: implementations, which is #2961's subject and not something a shared constant
#: can fix.
UNDATABLE_ROW_FALLBACK = _GROUPING_EPOCH

#: Exactly the strings SQLite's ``julianday`` can read, which is what the
#: canonical order compares. Written as the WHOLE value, not a prefix: the first
#: version of this guard checked only the date, and the divergence between
#: Python and ``julianday`` is mostly in the time and the offset. Measured
#: against sqlite 3.50.4 over a battery of spellings — `+0500`, `-05`, a
#: lowercase `t`, a seconds-bearing offset, the basic form — this agrees with
#: the engine on every one, in both directions.
#:
#: The engine takes: a date; optionally a time after a space or an UPPERCASE
#: ``T``; optionally fractional seconds; optionally ``Z`` or ``+/-HH:MM``.
_JULIANDAY_READABLE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"          # date
    r"([ T]\d{2}:\d{2}"             # optional time, space or uppercase T
    r"(:\d{2}(\.\d+)?)?"           # optional seconds, optional fraction
    # NESTED inside the time, not beside it. Spelled as a sibling the two were
    # independently optional, so `2026-01-01Z` and `2026-01-01+01:00` matched —
    # and `julianday` returns NULL for a date carrying an offset but no time,
    # while the parser below dates them happily. That is exactly the
    # divergence this expression exists to close, reintroduced by where a
    # bracket fell.
    r"(Z|[+-]\d{2}:\d{2})?)?$"      # optional UTC marker or +/-HH:MM offset
)


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
        now: the stamp substituted for a row whose ``created_at`` is missing or
            unparseable. Defaults to the stamp of the row BEFORE it (the epoch,
            for a transcript that begins with one) — deliberately not the wall
            clock. A wall clock made grouping a function of *when it was asked*:
            the same transcript grouped one way now and another way an hour
            later, because a bad row kept sliding forward and rejoining
            whichever session was newest. It also made the #2959 projection
            unable to cache this result, since a cache has to be reproducible
            from what it caches. Still injectable, and an injected value still
            wins, which is what the tests use.
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
            ``preview_wake_source`` (the first autonomous-wake source seen, so
            a session with no human turn can still be titled honestly),
            and — only when ``collect_messages`` — ``messages``.
        Callers reverse / slice / decorate as needed.
    """
    # The stamp last used, so an undatable row inherits the one before it.
    # LOCAL on purpose. An earlier fix made the substitute the transcript's
    # MAXIMUM stamp, which is deterministic but global: appending a row to one
    # session then re-dated an undatable row in a different, untouched session,
    # so an incremental repair that recomputed only the appended session left
    # the other stale and recorded a current watermark over it. A row-local rule
    # has no such coupling — a row's stamp depends on what precedes it, and
    # appending never changes that.
    previous: Optional[datetime] = None

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
            "preview_wake_source": None,
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

        # No wall-clock arm anywhere: every fallback here is a function of the
        # rows (or of a clock the caller injected deliberately), which is what
        # lets the #2959 projection cache this result at all.
        timestamp = (
            coerce_session_timestamp(msg.get("created_at"))
            or coerce_session_timestamp(now)
            or previous
            or _GROUPING_EPOCH
        )
        previous = timestamp

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
            # Operator-signal fallback notices (#operator_signals.py) and
            # autonomous signal wakes (#2204) are both persisted with
            # role="user" so they replay in history, but neither is something
            # the user typed — skip both when picking the preview so an
            # auto-mode/budget/governance notice or a `talon.job_complete`
            # wake never becomes the conversation's title (#2455, #2947).
            # Memory retrieval already treats the two flags as one class
            # (memory_retriever.py); the preview picker had only learned half
            # the pair.
            wake_source = signal_wake_source(meta)
            if wake_source is not None and current["preview_wake_source"] is None:
                # Remember the first wake so a session with NO human turn can
                # be labeled for the autonomous work it actually ran, instead
                # of falling through to an empty "New conversation" title.
                current["preview_wake_source"] = wake_source
            if (
                current["preview_content"] is None
                and not meta.get("operator_signal")
                and wake_source is None
            ):
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
        # only user rows were skipped operator-signal notices or autonomous
        # wakes, leaving its preview None. In that case fall through to a later
        # cluster's real preview so a resumed session never shows an empty
        # title. The earliest wake source travels with it, so a session that
        # never had a human turn is still labeled by the wake that started it.
        if existing["preview_content"] is None and session["preview_content"] is not None:
            existing["preview_content"] = session["preview_content"]
            existing["preview_metadata"] = session["preview_metadata"]
        if existing.get("preview_wake_source") is None:
            existing["preview_wake_source"] = session.get("preview_wake_source")
    return [merged[sid] for sid in order]


#: How a page of sessions is ordered, once. Newest activity first; ``session_id``
#: breaks ties.
#:
#: The tie-break is the point. Every caller used to sort on ``last_message_at``
#: alone and rely on Python's sort being *stable*, which silently made the
#: answer "whatever order grouping happened to emit". Ties are ordinary here —
#: SQLite stores history to the second, and a wake and the turn it triggers are
#: written in one transaction — so with a limit applied, which session appeared
#: on the page was decided by an implementation detail of the sort. Worse, that
#: rule cannot be expressed in SQL, so the #2959 projection could not reproduce
#: it and would have reordered tied sessions the day it replaced this path
#: (round-7 review).
#:
#: ``(column, descending)`` so the SQL clause and the Python sort are generated
#: from one declaration rather than written twice in two languages.
SESSION_ORDER: Tuple[Tuple[str, bool], ...] = (
    ("last_message_at", True),
    ("session_id", False),
)


def session_order_sql(backend_type: str) -> str:
    """:data:`SESSION_ORDER` as an ``ORDER BY`` clause.

    ``session_id`` is compared through :func:`bytewise_sql` so the SQL page and
    :func:`sort_sessions` break ties the same way. Without it the two agree only
    on a cluster whose default collation happens to be bytewise, which is not
    the common case.
    """
    return "ORDER BY " + ", ".join(
        f"{bytewise_sql(backend_type, column) if column == 'session_id' else column} "
        f"{'DESC' if descending else 'ASC'}"
        for column, descending in SESSION_ORDER
    )


def session_order_index_columns(backend_type: str) -> str:
    """:data:`SESSION_ORDER` as index columns, directions included.

    An index on the first key alone does not bound a page: ties on
    ``last_message_at`` are ordinary at second resolution, and the engine must
    then sort the whole tie group before applying ``LIMIT``. The tie-break is
    compared bytewise for the same reason the ``ORDER BY`` compares it that way,
    so the index and the ordering are the same comparison.
    """
    return ", ".join(
        f"{bytewise_sql(backend_type, column) if column == 'session_id' else column} "
        f"{'DESC' if descending else 'ASC'}"
        for column, descending in SESSION_ORDER
    )


def sort_sessions(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Order session dicts by :data:`SESSION_ORDER`, in place.

    Least significant key first, leaning on the sort being stable — which is
    the standard way to compose a multi-key ordering whose directions differ,
    and is safe precisely because no key is left implicit.
    """
    for column, descending in reversed(SESSION_ORDER):
        sessions.sort(key=lambda session: session[column], reverse=descending)
    return sessions


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
    sort_sessions(grouped)
    sessions = grouped[:limit]
    for session in sessions:
        preview = session.pop("preview_content", None) or ""
        session.pop("preview_metadata", None)
        wake_source = session.pop("preview_wake_source", None)
        if preview and preview_transform is not None:
            # e.g. unwrap sent-form so the preview is the raw user text.
            preview = preview_transform(preview)
        if preview:
            preview = preview[:preview_chars] + ("..." if len(preview) > preview_chars else "")
        elif wake_source:
            # No human turn in this session — it exists because a signal woke
            # the agent. Say so rather than handing the agent a blank title
            # (#2947); the UI twin applies the same label.
            preview = autonomous_wake_preview(wake_source)
        session["preview"] = preview
        session["is_trashed"] = include_trashed
        sid = session.get("session_id")
        if sid in names:
            session["name"] = names[sid]
    return sessions
