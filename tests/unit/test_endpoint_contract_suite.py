"""Focused contract tests for weak endpoint groups."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
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


def test_database_tables_endpoint_returns_shape_from_storage():
    db = MagicMock()
    db.backend_type = "sqlite"
    db.fetchall = AsyncMock(side_effect=[
        [("conversation_history",)],
        [(0, "id", "TEXT", 1, None, 1)],
    ])
    db.fetchone = AsyncMock(return_value=(3,))
    storage = MagicMock(db=db, db_path="/tmp/fake.db")
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/db/tables", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["table_count"] == 1
        assert payload["tables"][0]["name"] == "conversation_history"
        assert payload["tables"][0]["queryable"] is True
    finally:
        _restore_app(app, original)


def test_files_head_uses_existence_check_contract():
    file_store = MagicMock()
    file_store.file_exists = AsyncMock(return_value=True)
    file_store.get_file_metadata = AsyncMock(return_value=None)
    storage = MagicMock(files=file_store)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.head("/api/files/test-hash", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        assert response.headers["x-content-hash"] == "test-hash"
        assert response.headers["content-type"].startswith("application/octet-stream")
    finally:
        _restore_app(app, original)


def test_observability_events_endpoint_returns_serialized_events():
    event = SimpleNamespace(
        event_id="evt-1",
        timestamp="2026-03-15T00:00:00Z",
        agent_name="Claw",
        session_id="sess-1",
        event_type="tool_call",
        tool_name="search",
        duration_ms=12,
        success=True,
        error_message=None,
        metadata={"foo": "bar"},
    )
    agent = MagicMock(
        observability_store=MagicMock(query_events=AsyncMock(return_value=[event]))
    )

    app, original = _prepare_app(agent)
    # Observability router is now feature-contributed; mount it explicitly
    from endpoints.observability import router as obs_router
    app.include_router(obs_router)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/observability/events", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        assert response.json()["count"] == 1
        assert response.json()["events"][0]["event_id"] == "evt-1"
    finally:
        _restore_app(app, original)


def test_saved_items_structured_endpoint_uses_store_contract():
    agent = MagicMock(storage=MagicMock(db=MagicMock()), agent_id="did:agent")
    saved_item = MagicMock()
    saved_item.to_dict.return_value = {"id": "item-1", "schema_id": "contact"}

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with patch("kestrel_sovereign.storage.saved_items_store.SavedItemsStore.save_structured_item", AsyncMock(return_value=saved_item)):
                with TestClient(app) as client:
                    response = client.post(
                        "/api/saved-items/structured",
                        headers={"X-API-Key": "test-key"},
                        json={"schema_id": "contact", "content": {"name": "Ada"}},
                    )
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert response.json()["item"]["id"] == "item-1"
    finally:
        _restore_app(app, original)


def test_openai_compatible_endpoints_return_minimal_contracts():
    llm_service = MagicMock()
    llm_service.providers = [{"model": "gpt-5-mini"}]
    agent = MagicMock(llm_service=llm_service)
    agent.process_input = AsyncMock(return_value="hello")

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                models_response = client.get("/v1/models", headers={"X-API-Key": "test-key"})
                chat_response = client.post(
                    "/v1/chat/completions",
                    headers={"X-API-Key": "test-key"},
                    json={"model": "gpt-5-mini", "messages": [{"role": "user", "content": "hello"}]},
                )
        assert models_response.status_code == 200
        assert models_response.json()["object"] == "list"
        assert chat_response.status_code == 200
        assert chat_response.json()["object"] == "chat.completion"
        assert chat_response.json()["choices"][0]["message"]["content"] == "hello"
    finally:
        _restore_app(app, original)
