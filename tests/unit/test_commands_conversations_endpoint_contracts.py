"""Focused contract tests for commands and conversations endpoints."""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


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
    from endpoints.commands import BUILTIN_COMMANDS
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


def test_conversations_endpoint_groups_rows_and_marks_encrypted_preview():
    now = datetime(2026, 3, 17, 9, 0, 0)
    rows = [
        (4, "user", "plain text", "{}", now + timedelta(minutes=3)),
        (3, "system", "[New conversation started]", '{"new_session": true, "type": "session_marker"}', now + timedelta(minutes=2)),
        (2, "assistant", "hello", "{}", now + timedelta(minutes=1)),
        (1, "user", "ciphertext", '{"enc": true}', now),
    ]
    storage = MagicMock(agent_id="did:agent", encryption_enabled=True)
    storage.query_conversations = AsyncMock(return_value=rows)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch("endpoints.conversations.get_agent_fernet", return_value=object()):
            with patch("endpoints.conversations.decrypt_string", side_effect=ValueError("bad key")):
                with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                    with TestClient(app) as client:
                        response = client.get("/api/conversations?limit=10", headers=_api_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 2
        latest_session = payload["conversations"][0]
        older_session = payload["conversations"][1]
        assert latest_session["preview"] == "plain text"
        assert older_session["preview"] == "ciphertext"
        assert older_session["preview_encrypted"] is True
        assert payload["encrypted_at_rest"] is True
    finally:
        _restore_app(app, original)


def test_get_conversation_filters_session_markers_and_decrypts_messages():
    now = datetime(2026, 3, 17, 10, 0, 0)
    rows = [
        (10, "system", "[New conversation started]", '{"new_session": true, "type": "session_marker"}', now),
        (11, "user", "ciphertext", '{"enc": true}', now + timedelta(minutes=1)),
        (12, "assistant", "plain reply", "{}", now + timedelta(minutes=2)),
        (13, "system", "[New conversation started]", '{"new_session": true, "type": "session_marker"}', now + timedelta(minutes=3)),
    ]
    storage = MagicMock(agent_id="did:agent", encryption_enabled=True)
    storage.query_conversation_start = AsyncMock(return_value=(now,))
    storage.query_conversation_messages = AsyncMock(return_value=rows)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch("endpoints.conversations.get_agent_fernet", return_value=object()):
            with patch("endpoints.conversations.decrypt_string", return_value="decrypted text"):
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
    storage.query_conversation_start = AsyncMock(return_value=(now,))
    storage.query_conversation_messages = AsyncMock(return_value=rows)
    agent = MagicMock(storage=storage)

    def fake_decrypt(content, meta, fernet):
        return sent_form if content == "ciphertext-new" else legacy_raw

    app, original = _prepare_app(agent)
    try:
        with patch("endpoints.conversations.get_agent_fernet", return_value=object()):
            with patch("endpoints.conversations.decrypt_string", side_effect=fake_decrypt):
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


def test_list_conversations_includes_user_assigned_names():
    """List endpoint decorates sessions with their user-assigned ``name``
    when one is set; sessions without a rename don't get the key so the
    UI's ``conv.name || conv.preview`` fallback resolves to preview.

    Fixture matches the row-order contract the endpoint expects:
    ``query_conversations`` returns newest-first, and the endpoint reverses
    internally (see ``test_conversations_endpoint_groups_rows_and_marks_
    encrypted_preview`` for the canonical fixture shape).
    """
    now = datetime(2026, 3, 17, 9, 0, 0)
    rows = [
        # Newest-first.  Explicit new_session marker splits the two
        # clusters; session_ids emitted by the endpoint are the first
        # message id in each cluster (str-coerced).
        (4, "user", "second thread", "{}", now + timedelta(hours=2, minutes=1)),
        (3, "system", "[New conversation started]",
         '{"new_session": true, "type": "session_marker"}',
         now + timedelta(hours=2)),
        (2, "assistant", "hi there", "{}", now + timedelta(minutes=1)),
        (1, "user", "first thread", "{}", now),
    ]
    storage = MagicMock(agent_id="did:agent", encryption_enabled=False)
    storage.query_conversations = AsyncMock(return_value=rows)
    # Only the older session (first row id = "1") has a custom name.
    storage.get_conversation_names = AsyncMock(
        return_value={"1": "Custom Title"}
    )
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get(
                    "/api/conversations",
                    headers=_api_headers(),
                )
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 2
        conversations = payload["conversations"]
        named = [c for c in conversations if c.get("name") == "Custom Title"]
        assert len(named) == 1, (
            f"expected exactly one renamed session; got {conversations}"
        )
        # Un-renamed sessions must not have the key so the client's
        # ``conv.name || conv.preview`` fallback resolves to preview.
        un_renamed = [c for c in conversations if "name" not in c]
        assert un_renamed, "at least one un-named session expected"
        storage.get_conversation_names.assert_awaited_once()
    finally:
        _restore_app(app, original)


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
    storage.query_conversation_start = AsyncMock(return_value=(now,))
    storage.query_conversation_messages = AsyncMock(return_value=transcript_rows)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                new_response = client.post("/api/conversations/new", headers=_api_headers())
                delete_response = client.delete("/api/conversations/messages/21", headers=_api_headers())
                transcript_response = client.get(
                    "/api/conversations/20/transcript",
                    headers=_api_headers(),
                )
        assert new_response.status_code == 200
        assert new_response.json()["session_id"] == "20"
        storage.add_conversation.assert_awaited_once_with(
            role="system",
            content="[New conversation started]",
            metadata={"type": "session_marker", "new_session": True},
        )
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
                    headers=_api_headers(),
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
                    headers=_api_headers(),
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
                    headers=_api_headers(),
                )
        assert response.status_code == 200
        storage.delete_conversation_session.assert_awaited_once_with(
            "b5f0e218-12a4-4d6b-9e05-41b5adca7f6f",
            "did:agent",
        )
    finally:
        _restore_app(app, original)
