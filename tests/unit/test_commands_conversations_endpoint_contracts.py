"""Focused contract tests for commands and conversations endpoints."""

import time
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from kestrel_sovereign.storage.session_id_column import column_session_id


def _prepare_app(agent):
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
    }
    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    return app, original


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


def _api_headers():
    return {"X-API-Key": "test-key"}


def _destructive_headers():
    """Headers that satisfy the demo-isolation rail (#766) for live agents.

    The UI attaches X-Kestrel-Allow-Destructive automatically; tests
    that exercise destructive endpoints mirror that.
    """
    return {
        "X-API-Key": "test-key",
        "X-Kestrel-Allow-Destructive": "test-fixture",
    }


def test_commands_endpoint_merges_builtin_and_feature_commands():
    tool_schema = MagicMock(
        command_prefix="!ping",
        description="Ping the feature",
        parameters_summary="[target]",
    )
    tool = MagicMock(schema=tool_schema, name="ping")
    feature = MagicMock()
    feature.get_tools.return_value = [tool]
    agent = MagicMock(features={"signals": feature})

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/commands", headers=_api_headers())
        assert response.status_code == 200
        payload = response.json()
        commands = {entry["cmd"]: entry for entry in payload["commands"]}
        assert "!help" in commands
        assert commands["!ping"]["feature"] == "signals"
        assert commands["!ping"]["args"] == "[target]"
        assert payload["count"] == len(payload["commands"])
    finally:
        _restore_app(app, original)


def test_commands_endpoint_builtin_inventory_matches_command_handler_specs():
    from kestrel_sovereign.endpoints.commands import BUILTIN_COMMANDS
    from kestrel_sovereign.command_handler import BUILTIN_COMMAND_SPECS

    endpoint_cmds = {entry["cmd"] for entry in BUILTIN_COMMANDS}
    handler_cmds = {entry["cmd"] for entry in BUILTIN_COMMAND_SPECS}

    assert endpoint_cmds == handler_cmds
    assert "!sleep" in endpoint_cmds
    assert "!continue" in endpoint_cmds
    assert "!reload-context" in endpoint_cmds
    assert "!heartbeat" in endpoint_cmds


def test_sessions_endpoint_returns_message_totals_from_history():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "status?"},
    ]
    storage = MagicMock()
    storage.get_conversation_history = AsyncMock(return_value=history)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/sessions?limit=10", headers=_api_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload["messages"] == history
        assert payload["total"] == 3
        assert payload["user_messages"] == 2
        assert payload["agent_messages"] == 1
    finally:
        _restore_app(app, original)


LIST_AGENT = "did:test:conversation-list"


async def _seeded_list_storage(tmp_path, name, rows):
    """A real store + privacy wrapper holding ``rows``, for the list endpoint.

    The list reads the #2959 projection through the privacy layer (#2960), so a
    fixture that mocks the storage call proves nothing about where sessions come
    from. These go through the real derivation.

    The ``session_id`` COLUMN is derived from each row's metadata with
    ``column_session_id`` — the same function the write paths use — rather than
    left NULL for convenience. Leaving it NULL beside a stampable metadata id is
    a state no writer produces, and the projection refuses it on purpose; a
    fixture that manufactured it would be testing the refusal. Rows carrying no
    stampable id still land NULL, which is what legacy history looks like.

    ``rows`` are ``(role, content, metadata_json, created_at)``, oldest first.
    """
    storage = AsyncStorage(str(tmp_path / name))
    storage.agent_id = LIST_AGENT
    await storage.initialize()
    await storage.db.execute_many(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, session_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (LIST_AGENT, role, content, metadata,
             column_session_id(metadata), created_at)
            for role, content, metadata, created_at in rows
        ],
    )
    return storage, PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)


def _listed(app, query=""):
    with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
        with TestClient(app) as client:
            return client.get(f"/api/conversations{query}", headers=_api_headers())


@pytest.mark.asyncio
async def test_conversations_endpoint_groups_rows_and_marks_encrypted_preview(tmp_path):
    """Two clusters an hour apart list as two sessions, newest first, and an
    undecryptable preview is reported as encrypted rather than shown."""
    now = datetime(2026, 3, 17, 9, 0, 0)
    storage, wrapped = await _seeded_list_storage(tmp_path, "encrypted.db", [
        ("user", "ciphertext", '{"enc": true}', now),
        ("assistant", "hello", "{}", now + timedelta(minutes=1)),
        ("user", "plain text", "{}", now + timedelta(hours=2)),
    ])
    app, original = _prepare_app(MagicMock(storage=wrapped))
    try:
        with patch("kestrel_sovereign.endpoints.conversations.get_agent_fernet", return_value=object()):
            with patch("kestrel_sovereign.endpoints.conversations.decrypt_string", side_effect=ValueError("bad key")):
                response = _listed(app, "?limit=10")
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 2
        latest_session, older_session = payload["conversations"]
        assert latest_session["preview"] == "plain text"
        assert older_session["preview"] == "ciphertext"
        assert older_session["preview_encrypted"] is True
        # Nothing follows a page that held every session.
        assert payload["next_cursor"] is None
    finally:
        _restore_app(app, original)
        await storage.close()


def _wake_meta(session_id, source="talon.job_complete"):
    return json.dumps({
        "session_id": session_id,
        "signal_wake": {"source": source, "mode": "cognition"},
    })


@pytest.mark.asyncio
async def test_conversations_endpoint_skips_signal_wake_for_card_preview(tmp_path):
    """#2947: a COGNITION signal wake persists as role="user" so it replays in
    history, but it must not become the conversation card's title — the first
    real user turn does."""
    now = datetime(2026, 8, 10, 21, 0, 0)
    storage, wrapped = await _seeded_list_storage(tmp_path, "wake-skip.db", [
        ("user", "[TALON_JOB_COMPLETE] Background Talon job ...", _wake_meta("s1"), now),
        ("user", "what's the weather", json.dumps({"session_id": "s1"}), now + timedelta(minutes=1)),
        ("assistant", "sunny", json.dumps({"session_id": "s1"}), now + timedelta(minutes=2)),
    ])
    app, original = _prepare_app(MagicMock(storage=wrapped))
    try:
        response = _listed(app, "?limit=10")
        assert response.status_code == 200
        sessions = response.json()["conversations"]
        assert len(sessions) == 1
        assert sessions[0]["preview"] == "what's the weather"
    finally:
        _restore_app(app, original)
        await storage.close()


@pytest.mark.asyncio
async def test_conversations_endpoint_labels_a_wake_only_session_autonomous(tmp_path):
    """#2947: unattended dispatch and heartbeat-born sessions have a wake as
    their only user row. Skipping it must not leave an empty preview — the UI
    would render that as 'New conversation', a lie about a session that ran
    autonomous work."""
    now = datetime(2026, 8, 10, 21, 20, 0)
    storage, wrapped = await _seeded_list_storage(tmp_path, "wake-only.db", [
        ("user", "[TALON_JOB_COMPLETE] Background Talon job ...", _wake_meta("s1"), now),
        ("assistant", "reviewed the PR", json.dumps({"session_id": "s1"}), now + timedelta(minutes=1)),
    ])
    app, original = _prepare_app(MagicMock(storage=wrapped))
    try:
        response = _listed(app, "?limit=10")
        assert response.status_code == 200
        sessions = response.json()["conversations"]
        assert len(sessions) == 1
        assert sessions[0]["preview"] == "Autonomous wake — talon.job_complete"
        # Raw picker fields are consumed by the decorator, never serialized.
        assert "preview_wake_source" not in sessions[0]
        assert "preview_content" not in sessions[0]
    finally:
        _restore_app(app, original)
        await storage.close()


@pytest.mark.asyncio
async def test_conversations_endpoint_exposes_marker_uuid_as_session_id(tmp_path):
    """#2012: the list identifier must be the session's canonical UUID
    (metadata.session_id on the new_session marker), NOT the message
    row-id — so the value the UI round-trips matches where messages are
    filed and the pane doesn't load empty on a hard refresh."""
    now = datetime(2026, 6, 28, 9, 0, 0)
    uuid = "e1fd6fe5-885e-4d8b-9aaa-000000000099"
    storage, wrapped = await _seeded_list_storage(tmp_path, "marker-uuid.db", [
        ("system", "[New conversation started]",
         json.dumps({"new_session": True, "type": "session_marker", "session_id": uuid}), now),
        ("user", "hi", json.dumps({"session_id": uuid}), now + timedelta(minutes=1)),
        ("assistant", "hello", json.dumps({"session_id": uuid}), now + timedelta(minutes=2)),
    ])
    app, original = _prepare_app(MagicMock(storage=wrapped))
    try:
        response = _listed(app, "?limit=10")
        assert response.status_code == 200
        sessions = response.json()["conversations"]
        assert len(sessions) == 1
        # Identity is the UUID, not the marker's row-id.
        assert sessions[0]["session_id"] == uuid
    finally:
        _restore_app(app, original)
        await storage.close()


@pytest.mark.asyncio
async def test_conversations_endpoint_falls_back_to_rowid_for_legacy_cluster(tmp_path):
    """A genuinely legacy cluster with no session_id anywhere still keys by the
    first message's row-id, and is REACHABLE — #2960's projection could not hold
    such a key when Phase B shipped, and a list served from a table that drops
    them would have made those conversations vanish instead of merely rank low.

    Measured on Emma's live database: 473 of 1,522 live rows carry no session
    id, so this is the ordinary shape of old history, not an edge case.
    """
    now = datetime(2026, 6, 28, 11, 0, 0)
    storage, wrapped = await _seeded_list_storage(tmp_path, "legacy-rowid.db", [
        ("user", "first", "{}", now),
        ("assistant", "reply", "{}", now + timedelta(minutes=1)),
    ])
    app, original = _prepare_app(MagicMock(storage=wrapped))
    try:
        response = _listed(app, "?limit=10")
        assert response.status_code == 200
        sessions = response.json()["conversations"]
        assert [s["session_id"] for s in sessions] == ["1"]
        assert sessions[0]["preview"] == "first"
        # ...and the key opens. A listed session nobody can open would be a
        # different way of losing the conversation.
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                detail = client.get("/api/conversations/1", headers=_api_headers())
        assert detail.status_code == 200
        assert [m["content"] for m in detail.json()["messages"]] == ["first", "reply"]
    finally:
        _restore_app(app, original)
        await storage.close()


@pytest.mark.asyncio
async def test_a_session_id_the_indexed_column_cannot_hold_is_still_listed(tmp_path):
    """``rasa_shim`` files every SMS turn under ``sms:{sender}`` — core code, not
    a hypothetical — and a colon was outside Phase A's column charset, so the
    column stayed NULL for those rows for ever.

    The column is not what opens a session. ``_get_session_messages`` resolves a
    row id or matches ``metadata LIKE '%"session_id": "<value>"%'``, and an
    ``sms:`` session is found by the second. A list built on "could the column
    hold this key" therefore dropped a whole channel's conversations — the exact
    disappearance this ticket exists to end, reached by another route.

    Since #3061 the column CAN hold it, and that is asserted here too: a
    permanently NULL column is what kept an agent's every repair re-deriving its
    whole history. The listing claim above is unchanged and is the one this test
    is named for — it must hold whether the column is silent or not.
    """
    now = datetime(2026, 5, 1, 9, 0, 0)
    sms = json.dumps({"session_id": "sms:+15551234567"})
    storage, wrapped = await _seeded_list_storage(tmp_path, "sms.db", [
        ("user", "text me the forecast", sms, now),
        ("assistant", "clear all week", sms, now + timedelta(minutes=1)),
    ])
    app, original = _prepare_app(MagicMock(storage=wrapped))
    try:
        response = _listed(app, "?limit=10")
        assert response.status_code == 200
        sessions = response.json()["conversations"]
        assert [s["session_id"] for s in sessions] == ["sms:+15551234567"]
        assert sessions[0]["preview"] == "text me the forecast"
        assert sessions[0]["message_count"] == 2
        # ...and the column holds the id since #3061 widened the charset to
        # printable ASCII. The fixture stamps it the way a writer does, so this
        # is the state production is in, not one the test built.
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_history "
            "WHERE agent_id = ? AND session_id = ?",
            (LIST_AGENT, "sms:+15551234567"),
        ) == 2
    finally:
        _restore_app(app, original)
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_every_session_is_reachable_by_paging_a_history_past_the_old_window(
    tmp_path,
):
    """#2960's acceptance: a corpus larger than the retired 1,000-row window,
    and every session reachable by asking again.

    The path this replaces fetched ``min(limit * 20, 1000)`` rows of history and
    grouped whatever fell inside. 2,500 sessions of two rows each is 5,000 rows,
    so the old read could see at most 500 of them and no ``limit`` could reach
    the rest — measured on Emma at 34% unreachable. Paging must find all 2,500,
    each exactly once, and stop by saying so rather than by repeating itself.
    """
    start = datetime(2026, 6, 9, 8, 0, 0)
    rows = []
    for index in range(2500):
        session_start = start + timedelta(minutes=index * 40)
        rows.append(("user", f"preview {index}", "{}", session_start))
        rows.append(("assistant", f"reply {index}", "{}", session_start + timedelta(minutes=1)))
    storage, wrapped = await _seeded_list_storage(tmp_path, "deep-history.db", rows)
    assert len(rows) > 1000, "the corpus must exceed the window this ticket removed"

    app, original = _prepare_app(MagicMock(storage=wrapped))
    try:
        seen = []
        query = "?limit=500"
        pages = 0
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                while True:
                    response = client.get(f"/api/conversations{query}", headers=_api_headers())
                    assert response.status_code == 200
                    payload = response.json()
                    seen.extend(s["session_id"] for s in payload["conversations"])
                    pages += 1
                    assert pages <= 20, "paging did not terminate"
                    if not payload["next_cursor"]:
                        break
                    query = f"?limit=500&cursor={payload['next_cursor']}"

        assert len(seen) == len(set(seen)), "a session was served on two pages"
        assert len(seen) == 2500, f"{2500 - len(seen)} sessions were unreachable"
        # Newest-first, all the way down, across the page seams.
        assert seen[0] == "4999" and seen[-1] == "1"
    finally:
        _restore_app(app, original)
        await storage.close()


@pytest.mark.asyncio
async def test_a_half_built_index_is_a_503_not_a_truncated_list(tmp_path):
    """A partial projection is missing the FRONT of the list, not the tail.

    The walk goes forward by row id, so what a budget-exhausted repair has
    written is the OLDEST sessions — while the list is ordered by most recent
    activity. Serving it hands the user their oldest conversations, omits every
    recent one, and ends with ``next_cursor: null``, which reads as the end of
    a list that is missing its beginning.

    503 says the index is not ready. It clears itself: the next request
    continues the walk.
    """
    now = datetime(2026, 5, 1, 9, 0, 0)
    # Two sessions, so a page of one has a second page to continue to.
    storage, wrapped = await _seeded_list_storage(tmp_path, "half-built.db", [
        ("user", "hello", "{}", now),
        ("assistant", "hi", "{}", now + timedelta(minutes=1)),
        ("user", "later", "{}", now + timedelta(hours=3)),
        ("assistant", "and again", "{}", now + timedelta(hours=3, minutes=1)),
    ])
    app, original = _prepare_app(MagicMock(storage=wrapped))
    try:
        # A walk that can never land: every repair leaves the watermark short.
        from kestrel_sovereign.storage.conversation_sessions import SessionWatermark

        async def never_complete():
            return SessionWatermark("gen", True, 1, 1, 0, 999)

        with patch(
            "kestrel_sovereign.storage.conversation_sessions"
            ".ConversationSessionProjection.accounted",
            side_effect=never_complete,
        ):
            response = _listed(app, "?limit=10")
        assert response.status_code == 503
        assert "still being built" in response.json()["detail"]
        assert response.headers.get("Retry-After") == "5"

        # ...and an index that IS complete serves normally, so the refusal is
        # not simply refusing everything.
        first = _listed(app, "?limit=1")
        assert first.status_code == 200

        # A CONTINUATION is refused too. It does not repair — moving the ground
        # under a cursor is how a keyset page starts skipping rows — but a page
        # read while the projection is being rewalked ends early and says
        # `next_cursor: null`, which reads as the end of the list.
        token = first.json()["next_cursor"]
        assert token, "the fixture needs a second page for this to mean anything"
        with patch(
            "kestrel_sovereign.storage.conversation_sessions"
            ".ConversationSessionProjection.accounted",
            side_effect=never_complete,
        ):
            assert _listed(app, f"?limit=1&cursor={token}").status_code == 503
    finally:
        _restore_app(app, original)
        await storage.close()


@pytest.mark.asyncio
async def test_a_cursor_this_build_cannot_read_is_a_client_error(tmp_path):
    """A cursor is client-supplied text. Restarting at page one for an
    unreadable one answers a request for page nine with page one and looks like
    a list that forgot where it was; raising past the handler reports a typo as
    a server fault. It is a 400."""
    storage, wrapped = await _seeded_list_storage(tmp_path, "bad-cursor.db", [
        ("user", "hello", "{}", datetime(2026, 6, 28, 11, 0, 0)),
    ])
    app, original = _prepare_app(MagicMock(storage=wrapped))
    try:
        assert _listed(app, "?cursor=not-a-cursor").status_code == 400
        # ...and one minted for another view, which is served by other
        # machinery and ordered by keys this view never produced.
        first = _listed(app, "?limit=1")
        assert first.status_code == 200
        token = first.json()["next_cursor"]
        assert token is None or _listed(
            app, f"?limit=1&view=archived&cursor={token}"
        ).status_code == 400
    finally:
        _restore_app(app, original)
        await storage.close()


def test_get_conversation_filters_session_markers_and_decrypts_messages():
    now = datetime(2026, 3, 17, 10, 0, 0)
    rows = [
        (10, "system", "[New conversation started]", '{"new_session": true, "type": "session_marker"}', now),
        (11, "user", "ciphertext", '{"enc": true}', now + timedelta(minutes=1)),
        (12, "assistant", "plain reply", "{}", now + timedelta(minutes=2)),
        (13, "system", "[New conversation started]", '{"new_session": true, "type": "session_marker"}', now + timedelta(minutes=3)),
    ]
    storage = MagicMock(agent_id="did:agent", encryption_enabled=True)
    storage.query_session_rows = AsyncMock(return_value=rows)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch("kestrel_sovereign.endpoints.conversations.get_agent_fernet", return_value=object()):
            with patch("kestrel_sovereign.endpoints.conversations.decrypt_string", return_value="decrypted text"):
                with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                    with TestClient(app) as client:
                        response = client.get("/api/conversations/10", headers=_api_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload["message_count"] == 2
        assert payload["messages"][0]["content"] == "decrypted text"
        assert payload["messages"][0]["encrypted"] is False
        assert payload["messages"][1]["content"] == "plain reply"
        assert payload["encrypted_at_rest"] is True
    finally:
        _restore_app(app, original)


def test_get_conversation_filters_pre_tool_reasoning_from_metadata():
    now = datetime(2026, 6, 22, 12, 0, 0)
    metadata = {
        "pre_tool_reasoning": {
            "content": "I'll save that now.",
            "seam": "\n\n",
        },
        "tool_results": [{"tool_call_id": "tc1", "name": "save_fact"}],
    }
    rows = [
        (20, "assistant", "Saved after the tool confirmed it.", json.dumps(metadata), now),
    ]
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.query_session_rows = AsyncMock(return_value=rows)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/conversations/20", headers=_api_headers())
        assert response.status_code == 200
        message = response.json()["messages"][0]
        assert message["content"] == "Saved after the tool confirmed it."
        assert "pre_tool_reasoning" not in message["metadata"]
        assert message["metadata"]["tool_results"] == metadata["tool_results"]
    finally:
        _restore_app(app, original)


def test_get_conversation_surfaces_assistant_model_provider():
    now = datetime(2026, 6, 23, 10, 0, 0)
    rows = [
        (21, "user", "hello", "{}", now, None, None),
        (
            22,
            "assistant",
            "hi",
            "{}",
            now + timedelta(seconds=1),
            "gpt-5.5",
            "openai:plan",
        ),
    ]
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.query_session_rows = AsyncMock(return_value=rows)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/conversations/21", headers=_api_headers())
        assert response.status_code == 200
        messages = response.json()["messages"]
        assert "model" not in messages[0]
        assert messages[1]["model"] == "gpt-5.5"
        assert messages[1]["provider"] == "openai:plan"
    finally:
        _restore_app(app, original)


def test_get_conversation_unwraps_sent_form_user_content():
    """Rows written with metadata.sent_form=True carry the full rendered
    sent-form (<retrieved_context>.../<user_input>... wrappers). The
    detail endpoint MUST strip those wrappers so the chat UI shows raw
    user text — otherwise users see XML-looking tags in the log.

    Live-verified: before this, /api/conversations/{id} returned
    '<retrieved_context>...</retrieved_context>\\n<user_input>\\nhello...\\n</user_input>'
    as message.content. After, it returns just 'hello...'.
    """
    now = datetime(2026, 4, 24, 18, 25, 0)
    sent_form = (
        "<retrieved_context>\n<memories>\nM1\n</memories>\n"
        "</retrieved_context>\n<user_input>\nhello, are you really here?\n</user_input>"
    )
    legacy_raw = "raw from before sent-form existed"
    rows = [
        (
            415, "user", "ciphertext-new",
            '{"enc": true, "sent_form": true}',
            now,
        ),
        (
            414, "user", "ciphertext-legacy",
            '{"enc": true}',
            now + timedelta(minutes=1),
        ),
    ]
    storage = MagicMock(agent_id="did:agent", encryption_enabled=True)
    storage.query_session_rows = AsyncMock(return_value=rows)
    agent = MagicMock(storage=storage)

    def fake_decrypt(content, meta, fernet):
        return sent_form if content == "ciphertext-new" else legacy_raw

    app, original = _prepare_app(agent)
    try:
        with patch("kestrel_sovereign.endpoints.conversations.get_agent_fernet", return_value=object()):
            with patch("kestrel_sovereign.endpoints.conversations.decrypt_string", side_effect=fake_decrypt):
                with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                    with TestClient(app) as client:
                        response = client.get("/api/conversations/415", headers=_api_headers())
        assert response.status_code == 200
        messages = response.json()["messages"]
        new_msg = next(m for m in messages if m["id"] == 415)
        legacy_msg = next(m for m in messages if m["id"] == 414)
        # sent-form row → stripped back to raw user text
        assert new_msg["content"] == "hello, are you really here?"
        # legacy row (no flag) → pass-through as-is
        assert legacy_msg["content"] == legacy_raw
    finally:
        _restore_app(app, original)


def test_rename_conversation_happy_path_returns_stored_name():
    """PATCH /conversations/{id} with {"name": "new"} → 200 with the
    stored value echoed back.  Issue #716.
    """
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.set_conversation_name = AsyncMock(return_value="My Debugging Thread")
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.patch(
                    "/api/conversations/sess-abc",
                    json={"name": "  My Debugging Thread  "},
                    headers=_api_headers(),
                )
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["session_id"] == "sess-abc"
        assert payload["name"] == "My Debugging Thread"
        storage.set_conversation_name.assert_awaited_once_with(
            "sess-abc", "  My Debugging Thread  ",
        )
    finally:
        _restore_app(app, original)


def test_rename_conversation_empty_string_clears_name():
    """An empty / whitespace-only name clears the override — storage
    returns None and the endpoint surfaces name=null.
    """
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.set_conversation_name = AsyncMock(return_value=None)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.patch(
                    "/api/conversations/sess-abc",
                    json={"name": "   "},
                    headers=_api_headers(),
                )
        assert response.status_code == 200
        assert response.json()["name"] is None
        storage.set_conversation_name.assert_awaited_once_with("sess-abc", "   ")
    finally:
        _restore_app(app, original)


def test_rename_conversation_null_name_also_clears():
    """``{"name": null}`` is equivalent to clearing."""
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.set_conversation_name = AsyncMock(return_value=None)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.patch(
                    "/api/conversations/sess-abc",
                    json={"name": None},
                    headers=_api_headers(),
                )
        assert response.status_code == 200
        assert response.json()["name"] is None
        storage.set_conversation_name.assert_awaited_once_with("sess-abc", None)
    finally:
        _restore_app(app, original)


def test_rename_conversation_missing_field_400():
    """Body without a 'name' field → 400; storage never touched."""
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.set_conversation_name = AsyncMock()
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.patch(
                    "/api/conversations/sess-abc",
                    json={"not_name": "x"},
                    headers=_api_headers(),
                )
        assert response.status_code == 400
        storage.set_conversation_name.assert_not_awaited()
    finally:
        _restore_app(app, original)


def test_rename_conversation_non_string_name_400():
    """Non-string / non-null 'name' → 400."""
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.set_conversation_name = AsyncMock()
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.patch(
                    "/api/conversations/sess-abc",
                    json={"name": 42},
                    headers=_api_headers(),
                )
        assert response.status_code == 400
        storage.set_conversation_name.assert_not_awaited()
    finally:
        _restore_app(app, original)


@pytest.mark.asyncio
async def test_list_conversations_includes_user_assigned_names(tmp_path):
    """List endpoint decorates sessions with their user-assigned ``name``
    when one is set; sessions without a rename don't get the key so the
    UI's ``conv.name || conv.preview`` fallback resolves to preview."""
    now = datetime(2026, 3, 17, 9, 0, 0)
    storage, wrapped = await _seeded_list_storage(tmp_path, "names.db", [
        ("user", "first thread", "{}", now),
        ("assistant", "hi there", "{}", now + timedelta(minutes=1)),
        ("system", "[New conversation started]",
         '{"new_session": true, "type": "session_marker"}', now + timedelta(hours=2)),
        ("user", "second thread", "{}", now + timedelta(hours=2, minutes=1)),
    ])
    # Only the older session (keyed by its first row id) has a custom name.
    await storage.set_conversation_name("1", "Custom Title")

    app, original = _prepare_app(MagicMock(storage=wrapped))
    try:
        response = _listed(app)
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 2
        conversations = payload["conversations"]
        named = [c for c in conversations if c.get("name") == "Custom Title"]
        assert len(named) == 1, (
            f"expected exactly one renamed session; got {conversations}"
        )
        assert named[0]["session_id"] == "1"
        assert all("name" not in c for c in conversations if c["session_id"] != "1")
    finally:
        _restore_app(app, original)
        await storage.close()


@pytest.mark.asyncio
async def test_list_conversations_serves_the_archived_view_uncapped(tmp_path):
    """``view=archived`` (#2149) lists the sessions the user tidied away.

    Its membership — ``deleted_at IS NULL AND archived_at IS NOT NULL`` — is
    disjoint from the one the #2959 projection describes, so these are still
    derived by grouping. They are read WITHOUT a row cap: capping them is what
    put 34% of the active list out of reach, and the same cap behind the archive
    tab would leave the same defect standing there.
    """
    now = datetime(2026, 3, 17, 9, 0, 0)
    storage, wrapped = await _seeded_list_storage(tmp_path, "archived.db", [
        ("user", "live thread", "{}", now),
        ("assistant", "live reply", "{}", now + timedelta(minutes=1)),
    ])
    # A corpus of archived rows larger than the retired 1,000-row window.
    archived = []
    for index in range(600):
        started = now + timedelta(days=1, minutes=index * 40)
        archived.append((LIST_AGENT, "user", f"archived {index}", "{}", started, started))
        archived.append((LIST_AGENT, "assistant", f"reply {index}", "{}",
                         started + timedelta(minutes=1), started))
    await storage.db.execute_many(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, created_at, archived_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        archived,
    )

    app, original = _prepare_app(MagicMock(storage=wrapped))
    try:
        seen = []
        query = "?view=archived&limit=200"
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                for _ in range(10):
                    response = client.get(f"/api/conversations{query}", headers=_api_headers())
                    assert response.status_code == 200
                    payload = response.json()
                    seen.extend(s["session_id"] for s in payload["conversations"])
                    if not payload["next_cursor"]:
                        break
                    query = f"?view=archived&limit=200&cursor={payload['next_cursor']}"
        assert len(seen) == len(set(seen))
        assert len(seen) == 600, f"{600 - len(seen)} archived sessions were unreachable"
        # ...and the archived view never shows the live one.
        assert "1" not in seen
    finally:
        _restore_app(app, original)
        await storage.close()


def test_new_conversation_delete_message_and_transcript_contracts():
    now = datetime(2026, 3, 17, 11, 0, 0)
    transcript_rows = [
        (20, "system", "[New conversation started]", '{"new_session": true, "type": "session_marker"}', now),
        (21, "user", "hello", "{}", now + timedelta(minutes=1)),
        (22, "assistant", "summary", '{"type": "context_summary", "original_message_ids": [1, 2]}', now + timedelta(minutes=2)),
    ]
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.add_conversation = AsyncMock()
    storage.query_last_conversation_row = AsyncMock(return_value=(20, "2026-03-17T11:00:00"))
    storage.delete_conversation_message = AsyncMock(return_value=True)
    storage.query_session_rows = AsyncMock(return_value=transcript_rows)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                new_response = client.post("/api/conversations/new", headers=_api_headers())
                delete_response = client.delete(
                    "/api/conversations/messages/21",
                    headers=_destructive_headers(),
                )
                transcript_response = client.get(
                    "/api/conversations/20/transcript",
                    headers=_api_headers(),
                )
        assert new_response.status_code == 200
        # #2012: the response is the canonical UUID stamped on the marker (the
        # same id the list endpoint advertises and rename lands under), not the
        # marker row-id.
        new_sid = new_response.json()["session_id"]
        call = storage.add_conversation.await_args
        assert call.kwargs["role"] == "system"
        assert call.kwargs["content"] == "[New conversation started]"
        assert call.kwargs["metadata"] == {"type": "session_marker", "new_session": True}
        assert call.kwargs["session_id"] == new_sid
        assert new_sid and new_sid != "20"
        assert delete_response.status_code == 200
        assert delete_response.json() == {"success": True, "message_id": 21}
        assert transcript_response.status_code == 200
        assert transcript_response.headers["content-type"].startswith("text/markdown")
        transcript = transcript_response.text
        assert "# Conversation Transcript - Session 20" in transcript
        assert "**User**" in transcript
        assert "Type: context_summary" in transcript
        assert "Original messages: 1-2" in transcript
    finally:
        _restore_app(app, original)


def test_archive_and_unarchive_endpoint_contracts():
    """#2149: /archive and /unarchive delegate to storage and return the
    documented success envelope; a zero rowcount 404s."""
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.archive_conversation_session = AsyncMock(return_value=3)
    storage.unarchive_conversation_session = AsyncMock(return_value=3)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                archive_response = client.post(
                    "/api/conversations/sess-9/archive", headers=_api_headers()
                )
                unarchive_response = client.post(
                    "/api/conversations/sess-9/unarchive", headers=_api_headers()
                )
        assert archive_response.status_code == 200
        assert archive_response.json() == {
            "success": True,
            "session_id": "sess-9",
            "archived_count": 3,
        }
        storage.archive_conversation_session.assert_awaited_once_with(
            "sess-9", "did:agent"
        )
        assert unarchive_response.status_code == 200
        assert unarchive_response.json() == {
            "success": True,
            "session_id": "sess-9",
            "unarchived_count": 3,
        }
        storage.unarchive_conversation_session.assert_awaited_once_with(
            "sess-9", "did:agent"
        )
    finally:
        _restore_app(app, original)


def test_archive_endpoint_404s_on_zero_rows():
    """Archiving a session with no live rows returns 404 (matches restore)."""
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.archive_conversation_session = AsyncMock(return_value=0)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/conversations/ghost/archive", headers=_api_headers()
                )
        assert response.status_code == 404
    finally:
        _restore_app(app, original)


def test_get_conversation_marker_only_session_returns_empty_not_404():
    """#2012 (codex): a freshly started session has only a stripped marker, so
    the resolver yields zero rows — but the session EXISTS, so the detail
    endpoint must return 200 with an empty message list, not 404."""
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.query_session_rows = AsyncMock(return_value=[])
    storage.session_exists = AsyncMock(return_value=True)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get(
                    "/api/conversations/some-uuid", headers=_api_headers()
                )
        assert response.status_code == 200
        assert response.json()["messages"] == []
        assert response.json()["message_count"] == 0
    finally:
        _restore_app(app, original)


def test_get_conversation_missing_session_404s():
    """A session that doesn't exist at all still 404s."""
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.query_session_rows = AsyncMock(return_value=[])
    storage.session_exists = AsyncMock(return_value=False)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get(
                    "/api/conversations/ghost-uuid", headers=_api_headers()
                )
        assert response.status_code == 404
    finally:
        _restore_app(app, original)


def test_transcript_resolves_uuid_session_id():
    """#2012 (codex): the list API now advertises marker UUIDs, so the
    transcript endpoint must resolve a UUID session_id via the same
    dual-scheme resolver — not 404 because it only understood row-ids."""
    now = datetime(2026, 6, 28, 14, 0, 0)
    uuid = "e1fd6fe5-885e-4d8b-9aaa-0000000000ff"
    rows = [
        (50, "user", "hi", json.dumps({"session_id": uuid}), now),
        (51, "assistant", "hello", json.dumps({"session_id": uuid}), now + timedelta(minutes=1)),
    ]
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.query_session_rows = AsyncMock(return_value=rows)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get(
                    f"/api/conversations/{uuid}/transcript", headers=_api_headers()
                )
        assert response.status_code == 200
        text = response.text
        assert f"# Conversation Transcript - Session {uuid}" in text
        assert "hi" in text and "hello" in text
        storage.query_session_rows.assert_awaited_once_with(uuid, limit=1000)
    finally:
        _restore_app(app, original)


def test_delete_conversation_session_returns_200_with_count_when_messages_removed():
    """DELETE /conversations/{session_id} delegates to storage's
    delete_conversation_session and surfaces the row count for the UI
    (used by the "Conversation deleted (N messages)" toast).  Issue #715.
    """
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.delete_conversation_session = AsyncMock(return_value=7)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.delete(
                    "/api/conversations/session-xyz",
                    headers=_destructive_headers(),
                )
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["session_id"] == "session-xyz"
        assert payload["deleted_count"] == 7
        storage.delete_conversation_session.assert_awaited_once_with(
            "session-xyz", "did:agent",
        )
    finally:
        _restore_app(app, original)


def test_delete_conversation_session_returns_404_when_no_messages_deleted():
    """Storage returning 0 means the session didn't exist or was already
    empty; the endpoint surfaces that as 404 so the UI can roll back the
    optimistic row-fade-out.
    """
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.delete_conversation_session = AsyncMock(return_value=0)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.delete(
                    "/api/conversations/nonexistent",
                    headers=_destructive_headers(),
                )
        assert response.status_code == 404
    finally:
        _restore_app(app, original)


def test_delete_conversation_session_uuid_session_id_roundtrips_cleanly():
    """Session IDs in Kestrel are UUID4 strings
    (``async_conversation_store._new_session_id``), all safe for URLs
    without encoding.  Pin that a well-formed UUID round-trips through
    the route exactly to the storage layer — no truncation, no
    re-encoding, agent_id still attached.
    """
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.delete_conversation_session = AsyncMock(return_value=2)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.delete(
                    "/api/conversations/b5f0e218-12a4-4d6b-9e05-41b5adca7f6f",
                    headers=_destructive_headers(),
                )
        assert response.status_code == 200
        storage.delete_conversation_session.assert_awaited_once_with(
            "b5f0e218-12a4-4d6b-9e05-41b5adca7f6f",
            "did:agent",
        )
    finally:
        _restore_app(app, original)
