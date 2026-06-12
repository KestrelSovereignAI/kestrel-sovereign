"""Focused contract tests for remaining database/files/observability/saved-items routes."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


def _prepare_app(agent):
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    # ObservabilityFeature was moved to dynamic feature-mounting in #466
    # (`Feature.get_router()`), so its routes aren't on the core `app` until
    # an actual agent registers the feature at startup. This test uses a
    # MagicMock agent and bypasses the lifespan, so that mount never runs.
    # Mount the router once here, idempotently — without it, any test that
    # hits /api/observability/* gets a 404 unless another test on the same
    # pytest-xdist worker has already mounted it (race-y false-passes).
    _ensure_observability_router(app)

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
    }
    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    return app, original


def _ensure_observability_router(app):
    """Idempotently include endpoints/observability.py on ``app``.

    ``FastAPI.include_router`` would otherwise add duplicate routes on every
    call. Detect the presence of the summary route and skip if present."""
    target = "/api/observability/summary"
    for route in app.routes:
        if getattr(route, "path", None) == target:
            return
    from kestrel_sovereign.endpoints.observability import router as observability_router
    app.include_router(observability_router)


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


def _api_headers():
    return {"X-API-Key": "test-key"}


def test_database_table_query_contract_supports_search_and_pagination():
    db = MagicMock()
    db.backend_type = "sqlite"
    db.fetchall = AsyncMock(
        side_effect=[
            [(0, "id", "TEXT", 1, None, 1), (1, "content", "TEXT", 0, None, 0)],
            [("msg-1", "alpha " * 120)],
        ]
    )
    db.fetchone = AsyncMock(return_value=(3,))
    storage = MagicMock(db=db)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get(
                    "/api/db/tables/conversation_history?limit=1&offset=1&search=alp",
                    headers=_api_headers(),
                )
        assert response.status_code == 200
        payload = response.json()
        assert payload["table"] == "conversation_history"
        assert payload["columns"] == ["id", "content"]
        assert payload["total_rows"] == 3
        assert payload["has_more"] is True
        assert payload["rows"][0]["content"].endswith("...")
    finally:
        _restore_app(app, original)


def test_file_get_and_observability_summary_contracts():
    # The endpoint reads bytes through the privacy-wrapper facade
    # (storage.retrieve_file) so ISOLATED session-buffered files serve too
    # (#1662); MIME metadata still comes from the persistent files store.
    file_store = MagicMock()
    file_store.get_file_metadata = AsyncMock(return_value={"mime_type": "image/png"})
    event_error = SimpleNamespace(
        event_type="error",
        timestamp=datetime(2026, 3, 17, 14, 0, tzinfo=timezone.utc),
        metadata={"error_type": "tool_failed"},
        error_message="boom",
        duration_ms=None,
    )
    event_tool = SimpleNamespace(
        event_type="tool_response",
        timestamp=datetime(2026, 3, 17, 14, 1, tzinfo=timezone.utc),
        metadata=None,
        error_message=None,
        duration_ms=12,
    )
    observability_store = MagicMock(query_events=AsyncMock(return_value=[event_error, event_tool]))
    storage = MagicMock(files=file_store)
    storage.retrieve_file = AsyncMock(return_value=b"image-bytes")
    agent = MagicMock(storage=storage, observability_store=observability_store)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                file_response = client.get("/api/files/hash-1", headers=_api_headers())
                summary_response = client.get("/api/observability/summary?minutes=30", headers=_api_headers())
        assert file_response.status_code == 200
        assert file_response.headers["x-content-hash"] == "hash-1"
        assert file_response.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert file_response.headers["content-type"].startswith("image/png")
        assert file_response.content == b"image-bytes"
        assert summary_response.status_code == 200
        summary = summary_response.json()
        assert summary["time_window_minutes"] == 30
        assert summary["events_by_type"]["error"] == 1
        assert summary["events_by_type"]["tool_response"] == 1
        assert summary["error_count"] == 1
        assert summary["tool_responses_count"] == 1
        assert summary["avg_tool_duration_ms"] == 12.0
    finally:
        _restore_app(app, original)


def test_saved_items_listing_filters_and_schema_contracts():
    agent = MagicMock(storage=MagicMock(db=MagicMock(), agent_id="did:agent"), agent_id="did:agent")
    item = MagicMock()
    item.to_dict.return_value = {"id": "item-1", "item_type": "structured"}
    store = MagicMock()
    store.list_items = AsyncMock(return_value=[item])
    store.get_stats = AsyncMock(return_value={"total": 4, "by_type": {"structured": 4}})
    store.get_all_tags = AsyncMock(return_value=["urgent", "archive"])
    store.list_by_tag = AsyncMock(return_value=[item])
    store.list_by_schema = AsyncMock(return_value=[item])

    app, original = _prepare_app(agent)
    try:
        with patch("kestrel_sovereign.storage.saved_items_store.SavedItemsStore", return_value=store):
            with patch(
                "kestrel_sovereign.storage.saved_items_store.list_schemas",
                return_value=["contact", "recipe"],
            ):
                with patch(
                    "kestrel_sovereign.storage.saved_items_store.ITEM_SCHEMAS",
                    {"contact": {"fields": ["name"]}},
                ):
                    with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                        with TestClient(app) as client:
                            list_response = client.get("/api/saved-items?item_type=structured", headers=_api_headers())
                            stats_response = client.get("/api/saved-items/stats", headers=_api_headers())
                            schemas_response = client.get("/api/saved-items/schemas", headers=_api_headers())
                            tags_response = client.get("/api/saved-items/tags", headers=_api_headers())
                            by_tag_response = client.get("/api/saved-items/by-tag/urgent", headers=_api_headers())
                            by_schema_response = client.get("/api/saved-items/by-schema/contact", headers=_api_headers())
        assert list_response.status_code == 200
        assert list_response.json()["item_type_filter"] == "structured"
        assert stats_response.json()["by_type"]["structured"] == 4
        assert schemas_response.json()["schemas"] == ["contact", "recipe"]
        assert tags_response.json()["total"] == 2
        assert by_tag_response.json()["tag"] == "urgent"
        assert by_schema_response.json()["schema_id"] == "contact"
    finally:
        _restore_app(app, original)


def test_saved_items_endpoint_refuses_privacy_hidden_modes():
    from kestrel_sovereign.privacy import PrivacyConfig

    db = MagicMock()
    storage = MagicMock(db=db, agent_id="did:agent")
    agent = MagicMock(storage=storage, agent_id="did:agent")
    agent.privacy_config = PrivacyConfig(storage="none", llm_location="local")

    app, original = _prepare_app(agent)
    try:
        with patch("kestrel_sovereign.storage.saved_items_store.SavedItemsStore") as store_cls:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    response = client.post(
                        "/api/saved-items",
                        headers=_api_headers(),
                        json={
                            "item_type": "stash",
                            "name": "Do not persist",
                            "content": "EPHEMERAL_SECRET",
                        },
                    )
        assert response.status_code == 403
        assert "privacy mode" in response.json()["detail"]
        store_cls.assert_not_called()
    finally:
        _restore_app(app, original)


def test_saved_items_item_crud_search_and_pin_contracts():
    agent = MagicMock(storage=MagicMock(db=MagicMock(), agent_id="did:agent"), agent_id="did:agent")
    existing = MagicMock(ipfs_cid=None)
    existing.to_dict.return_value = {"id": "item-1", "name": "Ada"}
    updated = MagicMock()
    updated.to_dict.return_value = {"id": "item-1", "name": "Ada Lovelace"}
    created = MagicMock()
    created.to_dict.return_value = {"id": "item-2", "name": "New Item"}
    store = MagicMock()
    store.get_by_id = AsyncMock(side_effect=[existing, existing, existing])
    store.save_item = AsyncMock(return_value=created)
    store.search = AsyncMock(return_value=[{"id": "item-1", "score": 0.9}])
    store.update_item = AsyncMock(return_value=updated)
    store.pin_item_to_ipfs = AsyncMock(return_value="bafyitem")
    store.delete_item = AsyncMock()

    app, original = _prepare_app(agent)
    try:
        with patch("kestrel_sovereign.storage.saved_items_store.SavedItemsStore", return_value=store):
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    get_response = client.get("/api/saved-items/item-1", headers=_api_headers())
                    create_response = client.post(
                        "/api/saved-items",
                        headers=_api_headers(),
                        json={"item_type": "stash", "name": "New Item", "content": "hello"},
                    )
                    search_response = client.post(
                        "/api/saved-items/search",
                        headers=_api_headers(),
                        json={"query": "Ada", "limit": 5},
                    )
                    update_response = client.patch(
                        "/api/saved-items/item-1",
                        headers=_api_headers(),
                        json={"name": "Ada Lovelace"},
                    )
                    pin_response = client.post("/api/saved-items/item-1/pin", headers=_api_headers())
                    delete_response = client.delete("/api/saved-items/item-1", headers=_api_headers())
        assert get_response.status_code == 200
        assert get_response.json()["item"]["id"] == "item-1"
        assert create_response.status_code == 200
        assert create_response.json()["item"]["id"] == "item-2"
        assert search_response.json()["results"][0]["score"] == 0.9
        assert update_response.json()["item"]["name"] == "Ada Lovelace"
        assert pin_response.json()["ipfs_cid"] == "bafyitem"
        assert delete_response.json() == {"success": True, "deleted_id": "item-1"}
    finally:
        _restore_app(app, original)
