"""Focused contract tests for commands and conversations endpoints."""

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
