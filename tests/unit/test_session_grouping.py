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
    aware = datetime(2026, 6, 29, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5)))

    assert canonical_timestamp_sql("sqlite", "created_at") == "julianday(created_at)"
    assert timestamp_predicate("sqlite", "created_at", ">") == (
        "julianday(created_at) > julianday(?)"
    )
    assert timestamp_query_param("sqlite", aware) == "2026-06-29T12:00:00+00:00"

    postgres_bound = timestamp_query_param("postgres", aware)
    assert postgres_bound == datetime(2026, 6, 29, 12, 0, 0)
    assert postgres_bound.tzinfo is None
    assert timestamp_predicate("postgres", "created_at", ">") == "created_at > ?"


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


def test_the_parser_accepts_exactly_what_the_ordering_can_express():
    """One domain, not two that agree by accident.

    The canonical order compares SQLite timestamps through ``julianday``. If the
    parser accepts a form ``julianday`` cannot read, that row parses as a
    perfectly good timestamp everywhere in Python while sorting at the far end
    of the SQL order — where ``LIMIT`` can drop it out of the conversation list
    altogether.

    ``datetime.fromisoformat`` on Python 3.11+ accepts the BASIC form
    (``20260101T110000``), which no writer here produces and this function never
    documented. Accepting it was incidental permissiveness; the two domains are
    the same set now.
    """
    import itertools
    import sqlite3

    # GENERATED, not listed. The curated list below is still here for the
    # oddities no product of parts would produce, but the parts themselves are
    # enumerated — because the gap that got through was not an exotic spelling,
    # it was an ordinary date carrying an ordinary offset and no time, and no
    # hand-written corpus happened to contain one. A list can only fail to
    # include something; a product cannot.
    generated = [
        date + time + zone
        for date, time, zone in itertools.product(
            ("2026-01-01",),
            ("", " 11:00", "T11:00", " 11:00:00", "T11:00:00", "T11:00:00.123456"),
            ("", "Z", "+01:00", "-05:00"),
        )
    ]
    assert "2026-01-01+01:00" in generated and "2026-01-01Z" in generated, (
        "the product stopped covering a bare date with an offset, which is the "
        "form this case was extended for"
    )

    db = sqlite3.connect(":memory:")
    try:
        for value in generated + [
            # Readable by both.
            "2026-01-01",
            "2026-01-01 11:00",
            "2026-01-01 10:00:00",
            "2026-01-01T10:00:00",
            "2026-01-01 10:00:00.123456",
            "2026-01-01T11:00:00+00:00",
            "2026-01-01T11:00:00-05:00",
            "2026-01-01T11:00:00.123+02:00",
            "2026-01-01T11:00:00Z",
            # Readable by Python alone, which is the whole point. The first
            # version of this guard checked only the DATE prefix and let every
            # one of these through — the divergence lives in the time and the
            # offset. The lowercase `t` needed the gate moved ahead of
            # `strptime` too, which compiles its format with `re.IGNORECASE`.
            "20260101T110000",
            "2026-01-01T11:00:00+0500",
            "2026-01-01T11:00:00-05",
            "2026-01-01t11:00:00",
            "2026-01-01T11:00:00+00:00:30",
            # Readable by neither.
            "not-a-date",
            "",
        ]:
            sql = db.execute("SELECT julianday(?)", (value,)).fetchone()[0]
            python = coerce_session_timestamp(value)
            assert (sql is None) == (python is None), (
                f"{value!r}: julianday reads it as {sql!r} and the parser as "
                f"{python!r}. One of them orders this row and the other dates "
                "it, so they have to agree about whether it can be read at all."
            )
    finally:
        db.close()
