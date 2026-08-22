"""Unit tests for the shared session-boundary algorithm (#2019).

This helper is the single source of truth for how a flat message list becomes
sessions, used by BOTH the /api/conversations endpoint and the agent's
list_conversations memory tool. If it drifts, the UI and the agent disagree on
session boundaries and the agent can soft-delete the wrong conversation.
"""
from datetime import datetime, timedelta, timezone

from kestrel_sovereign.storage.session_grouping import (
    autonomous_wake_preview,
    canonical_timestamp_sql,
    coerce_session_timestamp,
    coalesce_sessions_by_session_id,
    group_messages_into_sessions,
    signal_wake_source,
    summarize_sessions,
    timestamp_predicate,
    timestamp_query_param,
)

BASE = datetime(2026, 6, 29, 12, 0, 0)


def _msg(i, role, content, *, minutes=0, session_id=None, new_session=False,
         operator_signal=False, signal_wake=None):
    meta = {}
    if session_id is not None:
        meta["session_id"] = session_id
    if new_session:
        meta["new_session"] = True
    if operator_signal:
        meta["operator_signal"] = True
    if signal_wake is not None:
        meta["signal_wake"] = signal_wake
    return {
        "id": i,
        "role": role,
        "content": content,
        "metadata": meta,
        "created_at": BASE + timedelta(minutes=minutes),
    }


def test_quick_succession_is_one_session():
    msgs = [
        _msg(1, "user", "hi", minutes=0, session_id="s1"),
        _msg(2, "assistant", "hello", minutes=1, session_id="s1"),
        _msg(3, "user", "more", minutes=2, session_id="s1"),
    ]
    sessions = group_messages_into_sessions(msgs)
    assert len(sessions) == 1
    s = sessions[0]
    assert s["session_id"] == "s1"
    assert s["message_count"] == 3
    assert s["user_message_count"] == 2
    assert s["preview_content"] == "hi"  # first user message


def test_time_gap_splits_sessions():
    msgs = [
        _msg(1, "user", "first convo", minutes=0, session_id="s1"),
        _msg(2, "assistant", "ok", minutes=1, session_id="s1"),
        # > 30 min later: a new session
        _msg(3, "user", "second convo", minutes=100, session_id="s2"),
    ]
    sessions = group_messages_into_sessions(msgs)
    assert [s["session_id"] for s in sessions] == ["s1", "s2"]
    assert sessions[0]["message_count"] == 2
    assert sessions[1]["message_count"] == 1
    assert sessions[1]["preview_content"] == "second convo"


def test_new_session_marker_splits_and_is_not_counted():
    msgs = [
        _msg(1, "user", "a", minutes=0, session_id="s1"),
        # explicit marker only 2 min later — gap alone wouldn't split
        _msg(2, "user", "", minutes=2, session_id="s2", new_session=True),
        _msg(3, "user", "b", minutes=3, session_id="s2"),
    ]
    sessions = group_messages_into_sessions(msgs)
    assert len(sessions) == 2
    # The marker row itself is structural and must not inflate the count.
    assert sessions[1]["session_id"] == "s2"
    assert sessions[1]["message_count"] == 1
    assert sessions[1]["preview_content"] == "b"


def test_session_id_change_splits_even_within_gap():
    # Distinct session_ids in adjacent turns (no gap, no marker) must split, so
    # each summary's id matches exactly its own rows (#2019).
    msgs = [
        _msg(1, "user", "first", minutes=0, session_id="s1"),
        _msg(2, "user", "second", minutes=1, session_id="s2"),
    ]
    sessions = group_messages_into_sessions(msgs)
    assert [s["session_id"] for s in sessions] == ["s1", "s2"]
    assert all(s["message_count"] == 1 for s in sessions)


def test_legacy_then_uuid_stays_merged_to_match_resolver():
    # A legacy row (no session_id) followed within the gap by a UUID row must
    # stay ONE session: _get_session_messages(<legacy row id>) time-walks
    # through the UUID row, so splitting the list there would let a legacy
    # delete also destroy the UUID session (#2019). Two distinct UUIDs still
    # split — see test_session_id_change_splits_even_within_gap.
    msgs = [
        _msg(1, "user", "legacy", minutes=0),               # no session_id
        _msg(2, "user", "stamped", minutes=1, session_id="u-1234"),
    ]
    sessions = group_messages_into_sessions(msgs)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "1"  # legacy anchor wins
    assert sessions[0]["message_count"] == 2


def test_unlabeled_turn_stays_with_current_session():
    # A turn with no session_id does not trigger a split.
    msgs = [
        _msg(1, "user", "a", minutes=0, session_id="s1"),
        _msg(2, "assistant", "b", minutes=1),  # no session_id
    ]
    sessions = group_messages_into_sessions(msgs)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "s1"
    assert sessions[0]["message_count"] == 2


def test_legacy_cluster_without_metadata_falls_back_to_row_id():
    msgs = [
        _msg(1, "user", "legacy", minutes=0),  # no session_id in metadata
        _msg(2, "assistant", "reply", minutes=1),
    ]
    sessions = group_messages_into_sessions(msgs)
    assert len(sessions) == 1
    # Falls back to the first message's row id, as a string.
    assert sessions[0]["session_id"] == "1"


def test_integer_session_id_is_treated_as_legacy_not_canonical():
    # A bare-integer session_id is a mis-filed key (#2012); the grouper must
    # not adopt it as the canonical label — it uses the row-id fallback.
    msgs = [_msg(7, "user", "x", minutes=0, session_id="123")]
    sessions = group_messages_into_sessions(msgs)
    assert sessions[0]["session_id"] == "7"


def test_unparseable_timestamp_uses_injected_clock():
    msgs = [
        {"id": 1, "role": "user", "content": "x", "metadata": {"session_id": "s1"},
         "created_at": "not-a-date"},
    ]
    sessions = group_messages_into_sessions(msgs, now=BASE)
    assert len(sessions) == 1
    assert sessions[0]["started_at"] == BASE.isoformat()


def test_timestamp_coercion_normalizes_iso_offsets_to_naive_utc():
    expected = datetime(2026, 6, 29, 12, 0, 0)

    assert coerce_session_timestamp("2026-06-29 12:00:00") == expected
    assert coerce_session_timestamp("2026-06-29T12:00:00Z") == expected
    assert coerce_session_timestamp("2026-06-29T13:00:00+01:00") == expected
    assert coerce_session_timestamp(
        datetime(2026, 6, 29, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
    ) == expected
    assert coerce_session_timestamp("not-a-date") is None


def test_timestamp_sql_boundary_normalizes_sqlite_and_preserves_postgres_binds():
    """Both sides of a timestamp comparison, in the spellings each backend uses.

    The SQLite parameter used to be ``value.isoformat()`` — a ``T`` separator,
    a ``+00:00`` offset — which was safe only because ``julianday`` was applied
    to BOTH sides and reads either. Since #3009 the column holds exactly one
    spelling, and the parameter is rendered in it: a bound value in a different
    spelling would be the one term in the comparison still needing conversion,
    which is what has to stop being true before ``julianday`` can come out of
    the ordering at all.
    """
    aware = datetime(2026, 6, 29, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5)))

    # The column compares as ITSELF on both backends now — the whole of step 5
    # in one assertion. A function call around it is not indexable, which is
    # why removing it is the point rather than a tidy-up.
    assert canonical_timestamp_sql("sqlite", "created_at") == "created_at"
    assert timestamp_predicate("sqlite", "created_at", ">") == "created_at > ?"
    assert timestamp_query_param("sqlite", aware) == "2026-06-29 12:00:00"
    # The offset is APPLIED on the way, not dropped: 07:00-05:00 is 12:00 UTC.
    assert timestamp_query_param("sqlite", "2026-06-29T07:00:00-05:00") == (
        "2026-06-29 12:00:00"
    )
    # A value nothing can date is passed through rather than becoming NULL: a
    # predicate against it matched nothing before and must go on matching
    # nothing, instead of comparing NULL against every row.
    assert timestamp_query_param("sqlite", "not a date") == "not a date"
    postgres_bound = timestamp_query_param("postgres", aware)
    assert postgres_bound == datetime(2026, 6, 29, 12, 0, 0)
    assert postgres_bound.tzinfo is None
    assert timestamp_predicate("postgres", "created_at", ">") == "created_at > ?"


def test_a_fractional_boundary_is_not_rounded_into_a_row_it_excludes():
    """Sub-second precision survives the bind, because a purge compares on it.

    Stored values are whole seconds — no writer produces a fraction and #3009's
    migration truncates them — but a caller's BOUNDARY may carry one, and the
    canonical spelling would round it down. ``purge_all_since`` compares
    ``>=``, so a boundary of ``12:00:00.500`` truncated to ``12:00:00`` starts
    selecting the row stamped ``12:00:00``, which predates it, and PERMANENTLY
    deletes it. PostgreSQL keeps the fraction, so the two backends would also
    disagree about what a purge destroys.

    Asked of the real predicate rather than of the formatting, because it is
    the comparison that decides which rows die.
    """
    import sqlite3

    engine = sqlite3.connect(":memory:")
    engine.execute("CREATE TABLE t (created_at TEXT)")
    engine.executemany(
        "INSERT INTO t VALUES (?)",
        [("2026-01-01 12:00:00",), ("2026-01-01 12:00:01",)],
    )
    predicate = timestamp_predicate("sqlite", "created_at", ">=")
    doomed = [
        row[0]
        for row in engine.execute(
            f"SELECT created_at FROM t WHERE {predicate}",
            (timestamp_query_param("sqlite", "2026-01-01 12:00:00.500"),),
        )
    ]
    assert doomed == ["2026-01-01 12:00:01"], (
        "a row stamped before the boundary was selected for permanent "
        "deletion, because the boundary was rounded down to meet it"
    )


def test_grouping_safely_mixes_naive_and_aware_timestamp_shapes():
    msgs = [
        {
            "id": 1,
            "role": "user",
            "content": "first",
            "metadata": {},
            "created_at": "2026-06-29 12:00:00",
        },
        {
            "id": 2,
            "role": "assistant",
            "content": "second",
            "metadata": {},
            "created_at": "2026-06-29T13:01:00+01:00",
        },
    ]

    sessions = group_messages_into_sessions(msgs)

    assert len(sessions) == 1
    assert sessions[0]["message_count"] == 2


def test_empty_input_returns_no_sessions():
    assert group_messages_into_sessions([]) == []


def test_coalesce_merges_resumed_same_uuid_clusters():
    # A conversation resumed past the gap re-supplies the same UUID, so the
    # grouper emits two clusters with an identical session_id. Coalescing must
    # fold them into one unique delete target (#2019).
    msgs = [
        _msg(1, "user", "earlier", minutes=0, session_id="s-resumed"),
        _msg(2, "assistant", "ok", minutes=1, session_id="s-resumed"),
        _msg(3, "user", "much later", minutes=200, session_id="s-resumed"),
    ]
    grouped = group_messages_into_sessions(msgs)
    assert len(grouped) == 2  # split by the gap...
    coalesced = coalesce_sessions_by_session_id(grouped)
    assert len(coalesced) == 1  # ...but one session_id, one target
    s = coalesced[0]
    assert s["session_id"] == "s-resumed"
    assert s["message_count"] == 3
    assert s["user_message_count"] == 2
    assert s["preview_content"] == "earlier"  # earliest cluster's preview


def test_operator_signal_notice_is_skipped_for_preview():
    # A route that can't take inline system messages persists the operator
    # notice (auto-mode/budget/governance) as role="user" so it replays, but
    # it must never win the preview slot over the real first user message.
    msgs = [
        _msg(1, "user", "<operator_notice>...</operator_notice>", minutes=0,
             session_id="s1", operator_signal=True),
        _msg(2, "user", "what's the weather", minutes=0, session_id="s1"),
        _msg(3, "assistant", "sunny", minutes=1, session_id="s1"),
    ]
    sessions = group_messages_into_sessions(msgs)
    assert len(sessions) == 1
    assert sessions[0]["preview_content"] == "what's the weather"
    # Still counted as real turns for message_count/user_message_count.
    assert sessions[0]["user_message_count"] == 2


def test_operator_signal_only_session_has_no_preview():
    msgs = [
        _msg(1, "user", "<operator_notice>...</operator_notice>", minutes=0,
             session_id="s1", operator_signal=True),
    ]
    sessions = group_messages_into_sessions(msgs)
    assert len(sessions) == 1
    assert sessions[0]["preview_content"] is None


def test_coalesce_backfills_preview_when_first_cluster_is_operator_only():
    # First cluster's only user row is a skipped operator notice (no real
    # preview); the session resumes past the gap with a real user message.
    # Coalescing must pull that real preview forward instead of surfacing an
    # empty title (codex review follow-up).
    msgs = [
        _msg(1, "user", "<operator_notice>...</operator_notice>", minutes=0,
             session_id="s-resumed", operator_signal=True),
        _msg(2, "user", "real question", minutes=200, session_id="s-resumed"),
    ]
    grouped = group_messages_into_sessions(msgs)
    assert len(grouped) == 2  # split by the gap
    assert grouped[0]["preview_content"] is None
    coalesced = coalesce_sessions_by_session_id(grouped)
    assert len(coalesced) == 1
    assert coalesced[0]["preview_content"] == "real question"


WAKE = {"source": "talon.job_complete", "mode": "cognition"}


def test_signal_wake_is_skipped_for_preview():
    # A COGNITION signal wake (heartbeat, talon.job_complete, wait.complete,
    # restart.completed, a2a) persists as role="user" so it replays in history
    # (#2204), but it is synthetic — it must never win the preview slot over
    # the real user message (#2947).
    msgs = [
        _msg(1, "user", "[TALON_JOB_COMPLETE] Background Talon job ...", minutes=0,
             session_id="s1", signal_wake=WAKE),
        _msg(2, "user", "what's the weather", minutes=1, session_id="s1"),
        _msg(3, "assistant", "sunny", minutes=2, session_id="s1"),
    ]
    sessions = group_messages_into_sessions(msgs)
    assert len(sessions) == 1
    assert sessions[0]["preview_content"] == "what's the weather"
    # Still counted as real turns for message_count/user_message_count.
    assert sessions[0]["user_message_count"] == 2


def test_signal_wake_only_session_is_labeled_by_its_wake():
    # Unattended dispatch (CLI, scheduler, detached) and heartbeat-born
    # sessions have a wake as their FIRST user row and may never get a human
    # turn. That session has no preview_content, but it is not blank either —
    # it carries the wake source so callers can title it honestly (#2947).
    msgs = [
        _msg(1, "user", "[TALON_JOB_COMPLETE] Background Talon job ...", minutes=0,
             session_id="s1", signal_wake=WAKE),
        _msg(2, "assistant", "reviewed the PR", minutes=1, session_id="s1"),
    ]
    sessions = group_messages_into_sessions(msgs)
    assert len(sessions) == 1
    assert sessions[0]["preview_content"] is None
    assert sessions[0]["preview_wake_source"] == "talon.job_complete"


def test_signal_wake_without_source_still_marks_the_session_autonomous():
    msgs = [_msg(1, "user", "wake", minutes=0, session_id="s1", signal_wake=True)]
    sessions = group_messages_into_sessions(msgs)
    assert sessions[0]["preview_content"] is None
    assert sessions[0]["preview_wake_source"] == "signal"


def test_first_wake_source_wins_over_later_wakes():
    msgs = [
        _msg(1, "user", "heartbeat wake", minutes=0, session_id="s1",
             signal_wake={"source": "heartbeat", "mode": "cognition"}),
        _msg(2, "user", "talon wake", minutes=1, session_id="s1", signal_wake=WAKE),
    ]
    sessions = group_messages_into_sessions(msgs)
    assert sessions[0]["preview_wake_source"] == "heartbeat"


def test_ordinary_rows_carry_no_wake_source():
    msgs = [_msg(1, "user", "hi", minutes=0, session_id="s1")]
    sessions = group_messages_into_sessions(msgs)
    assert sessions[0]["preview_wake_source"] is None
    assert signal_wake_source({}) is None
    assert signal_wake_source({"signal_wake": WAKE}) == "talon.job_complete"


def test_coalesce_backfills_preview_when_first_cluster_is_wake_only():
    # Talon origin-session binding (#2877) lands wakes into a live session
    # across multi-hour gaps, so the grouper emits a wake-only first cluster.
    # Coalescing must surface the later real user message as the title while
    # keeping the wake source for sessions that never get one.
    msgs = [
        _msg(1, "user", "[TALON_JOB_COMPLETE] ...", minutes=0,
             session_id="s-resumed", signal_wake=WAKE),
        _msg(2, "user", "real question", minutes=200, session_id="s-resumed"),
    ]
    grouped = group_messages_into_sessions(msgs)
    assert len(grouped) == 2  # split by the gap
    assert grouped[0]["preview_content"] is None
    coalesced = coalesce_sessions_by_session_id(grouped)
    assert len(coalesced) == 1
    assert coalesced[0]["preview_content"] == "real question"
    assert coalesced[0]["preview_wake_source"] == "talon.job_complete"


def test_coalesce_carries_wake_source_forward_from_a_later_cluster():
    msgs = [
        _msg(1, "assistant", "orphan reply", minutes=0, session_id="s-resumed"),
        _msg(2, "user", "[TALON_JOB_COMPLETE] ...", minutes=200,
             session_id="s-resumed", signal_wake=WAKE),
    ]
    coalesced = coalesce_sessions_by_session_id(group_messages_into_sessions(msgs))
    assert len(coalesced) == 1
    assert coalesced[0]["preview_content"] is None
    assert coalesced[0]["preview_wake_source"] == "talon.job_complete"


def test_summarize_sessions_titles_a_wake_only_session():
    # The agent-facing twin must agree with the UI: a session whose only user
    # rows were wakes is named for the autonomous work it ran, never left
    # blank (which the UI would render as "New conversation") (#2947).
    msgs = [
        _msg(1, "user", "[TALON_JOB_COMPLETE] Background Talon job ...", minutes=0,
             session_id="s1", signal_wake=WAKE),
        _msg(2, "assistant", "reviewed the PR", minutes=1, session_id="s1"),
    ]
    summaries = summarize_sessions(msgs)
    assert len(summaries) == 1
    assert summaries[0]["preview"] == "Autonomous wake — talon.job_complete"
    assert summaries[0]["preview"] == autonomous_wake_preview("talon.job_complete")
    # Raw picker fields are consumed, never leaked to the tool payload.
    assert "preview_wake_source" not in summaries[0]


def test_summarize_sessions_prefers_a_real_user_message_over_the_wake_label():
    msgs = [
        _msg(1, "user", "[TALON_JOB_COMPLETE] Background Talon job ...", minutes=0,
             session_id="s1", signal_wake=WAKE),
        _msg(2, "user", "what's the weather", minutes=1, session_id="s1"),
    ]
    summaries = summarize_sessions(msgs)
    assert summaries[0]["preview"] == "what's the weather"


def test_summarize_sessions_never_labels_an_ordinary_session_autonomous():
    # No wake ran here, so the human turn is the title — the wake label must
    # not bleed onto ordinary conversations.
    msgs = [_msg(1, "user", "hi", minutes=0, session_id="s1")]
    summaries = summarize_sessions(msgs)
    assert summaries[0]["preview"] == "hi"


def test_coalesce_leaves_distinct_sessions_untouched():
    msgs = [
        _msg(1, "user", "a", minutes=0, session_id="s1"),
        _msg(2, "user", "b", minutes=200, session_id="s2"),
    ]
    coalesced = coalesce_sessions_by_session_id(group_messages_into_sessions(msgs))
    assert [s["session_id"] for s in coalesced] == ["s1", "s2"]


def test_an_undatable_row_is_dated_from_the_transcript_not_the_clock():
    """Grouping must be a function of the rows, not of when it was asked.

    ``now`` used to default to the wall clock, so a row with an unparseable
    ``created_at`` was dated to the moment of the call. Two consequences, both
    real: the same transcript grouped one way now and another way an hour
    later, as the bad row slid forward and kept rejoining whichever session was
    newest; and the #2959 projection could not cache the result, because a
    cache has to be reproducible from what it caches.

    The transcript here is historical, so a wall-clock default would date the
    bad row to today — months past every real stamp — and the gap rule would
    give it a session of its own. Pinning the newest stamp present keeps it
    with the run it was found in.
    """
    msgs = [
        _msg(1, "user", "hello", minutes=0, session_id="s1"),
        {"id": 2, "role": "assistant", "content": "undatable",
         "metadata": {"session_id": "s1"}, "created_at": "not-a-date"},
        _msg(3, "user", "still here", minutes=5, session_id="s1"),
    ]

    sessions = group_messages_into_sessions(msgs)

    assert len(sessions) == 1, (
        "the undatable row was dated far from the transcript and split off "
        f"into its own session: {[s['started_at'] for s in sessions]}"
    )
    assert sessions[0]["last_message_at"] == (BASE + timedelta(minutes=5)).isoformat()
    assert sessions[0]["message_count"] == 3


def test_grouping_an_undatable_row_is_repeatable():
    """The same transcript, grouped twice, must give the same answer.

    The wall-clock default made this false by construction — only by a few
    microseconds between two immediate calls, which is why asserting equality
    of two results catches it only when the value is compared exactly.
    """
    msgs = [
        _msg(1, "user", "hello", minutes=0, session_id="s1"),
        {"id": 2, "role": "assistant", "content": "undatable",
         "metadata": {"session_id": "s1"}, "created_at": None},
    ]

    assert group_messages_into_sessions(msgs) == group_messages_into_sessions(msgs)


def test_text_order_is_clock_order_for_everything_the_column_can_hold():
    """The claim that replaced "the parser and ``julianday`` agree" (#3009).

    This case used to require that ``coerce_session_timestamp`` accept exactly
    what SQLite's ``julianday`` could read, because the canonical order compared
    through ``julianday`` and a row the parser dated but SQL could not would
    sort at the far end of that order — where ``LIMIT`` drops it out of the
    conversation list.

    Step 5 removed the reason. ``created_at`` admits one fixed-width spelling
    now and the ordering compares the column as itself, so the requirement is
    no longer an agreement between two domains: it is that **text order is
    clock order** over the values the column can hold, and that the parser
    reads all of them. Two domains agreeing was always the weaker property —
    it had to be maintained. This one follows from the spelling.

    Still GENERATED rather than listed, for the reason the old case gave: the
    gap that got through last time was not an exotic spelling but an ordinary
    date carrying an ordinary offset, which no hand-written corpus happened to
    contain. A list can only fail to include something; a product cannot.
    """
    import itertools

    from kestrel_sovereign.storage.conversation_created_at import (
        canonical_created_at,
    )

    base = datetime(2026, 1, 1, 11, 0, 0)
    canonical = [
        canonical_created_at(base + timedelta(days=d, seconds=sec))
        for d, sec in itertools.product(
            (-400, -1, 0, 1, 400, 4000), (0, 1, 59, 3600, 86399)
        )
    ]
    assert len(set(canonical)) == len(canonical), "the corpus repeats itself"

    parsed = {value: coerce_session_timestamp(value) for value in canonical}
    unreadable = [v for v, dt in parsed.items() if dt is None]
    assert not unreadable, f"the parser cannot read stored values: {unreadable}"

    assert sorted(canonical) == sorted(canonical, key=lambda v: parsed[v]), (
        "lexicographic order diverged from chronological order, which is the "
        "property that lets the ordering compare created_at as itself"
    )


def test_a_predicate_compares_both_sides_the_same_way():
    """The two halves of a comparison must get the SAME treatment (#3009).

    Only ``created_at`` carries the canonical CHECK. ``deleted_at`` and
    ``archived_at`` are TIMESTAMP columns on the same table, written by the same
    code, and unconstrained — SQLite history really does hold both spellings in
    them — so they still have to be compared through ``julianday``.

    The placeholder is not a column and carries no guarantee of its own, so
    asking about it independently renders ``julianday(?)`` beside a bare
    ``created_at``: text compared against a float. SQLite answers that by TYPE
    order rather than by chronology and never raises, so a destructive
    predicate built that way silently matches the wrong set — and
    ``purge_trash_older_than`` is one of these.
    """
    import sqlite3

    assert timestamp_predicate("sqlite", "created_at", ">") == "created_at > ?"
    assert timestamp_predicate("sqlite", "deleted_at", ">") == (
        "julianday(deleted_at) > julianday(?)"
    )

    # And the mismatched form really does answer by type, not by clock.
    engine = sqlite3.connect(":memory:")
    mismatched = engine.execute(
        "SELECT '2026-06-01 12:00:00' > julianday('2026-06-01 11:00:00')"
    ).fetchone()[0]
    assert mismatched == 1, (
        "the premise died: SQLite no longer orders text above numbers, so a "
        "mismatched comparison would now fail loudly instead of silently"
    )
