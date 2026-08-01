"""Unit tests for the shared session-boundary algorithm (#2019).

This helper is the single source of truth for how a flat message list becomes
sessions, used by BOTH the /api/conversations endpoint and the agent's
list_conversations memory tool. If it drifts, the UI and the agent disagree on
session boundaries and the agent can soft-delete the wrong conversation.
"""
from datetime import datetime, timedelta, timezone

from kestrel_sovereign.storage.session_grouping import (
    canonical_timestamp_sql,
    coerce_session_timestamp,
    coalesce_sessions_by_session_id,
    group_messages_into_sessions,
    timestamp_predicate,
    timestamp_query_param,
)

BASE = datetime(2026, 6, 29, 12, 0, 0)


def _msg(i, role, content, *, minutes=0, session_id=None, new_session=False, operator_signal=False):
    meta = {}
    if session_id is not None:
        meta["session_id"] = session_id
    if new_session:
        meta["new_session"] = True
    if operator_signal:
        meta["operator_signal"] = True
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


def test_coalesce_leaves_distinct_sessions_untouched():
    msgs = [
        _msg(1, "user", "a", minutes=0, session_id="s1"),
        _msg(2, "user", "b", minutes=200, session_id="s2"),
    ]
    coalesced = coalesce_sessions_by_session_id(group_messages_into_sessions(msgs))
    assert [s["session_id"] for s in coalesced] == ["s1", "s2"]
