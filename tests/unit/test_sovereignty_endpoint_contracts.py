"""Focused contract tests for sovereignty endpoints."""

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


def test_storage_stats_returns_expected_shape(tmp_path):
    db_path = tmp_path / "agent.db"
    db_path.write_text("db")

    db = MagicMock()
    db.fetchone = AsyncMock(side_effect=[(4,), (2, 128)])
    db.fetchall = AsyncMock(return_value=[("agent", 1), ("memory", 3)])

    storage = MagicMock(db_path=str(db_path), db=db, agent_id="did:agent")
    storage.get_nodes_by_type = AsyncMock(
        side_effect=[
            [SimpleNamespace(node_id="exp-1")],
            [SimpleNamespace(node_id="bak-1"), SimpleNamespace(node_id="bak-2")],
        ]
    )
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/storage/stats", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["database"]["path"] == str(db_path)
        assert payload["conversations"]["count"] == 4
        assert payload["graph_nodes"]["agent"] == 1
        assert payload["files"]["count"] == 2
        assert payload["sovereignty_exports"] == 1
        assert payload["backups"] == 2
    finally:
        _restore_app(app, original)


def test_sovereignty_exports_serializes_receipts_and_backups():
    receipt = SimpleNamespace(
        node_id="exp-1",
        properties={
            "cid": "bafyexport",
            "storage_tier": "cloud_hot",
            "created_at": "2026-03-17T00:00:00Z",
            "encrypted": True,
        },
    )
    backup = SimpleNamespace(
        node_id="bak-1",
        properties={
            "ipfs_cid": "bafybackup",
            "storage_tier": "cloud_cold",
            "created_at": "2026-03-17T01:00:00Z",
            "encrypted": False,
        },
    )
    storage = MagicMock()
    storage.get_nodes_by_type = AsyncMock(side_effect=[[receipt], [backup]])
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/sovereignty/exports", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["exports"][0]["cid"] == "bafyexport"
        assert payload["backups"][0]["cid"] == "bafybackup"
    finally:
        _restore_app(app, original)


def test_sovereignty_export_rejects_invalid_tier():
    agent = MagicMock(storage=MagicMock())
    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/sovereignty/export",
                    headers={"X-API-Key": "test-key"},
                    json={"tier": "forbidden-tier"},
                )
        assert response.status_code == 400
        assert "Invalid tier" in response.json()["detail"]
    finally:
        _restore_app(app, original)


def test_sovereignty_import_rejects_invalid_cid():
    agent = MagicMock(storage=MagicMock())
    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/sovereignty/import",
                    headers={"X-API-Key": "test-key"},
                    json={"cid": "../bad-cid"},
                )
        assert response.status_code == 400
        assert "Invalid CID format" in response.json()["detail"]
    finally:
        _restore_app(app, original)


def test_sovereignty_files_listing_and_preview_contract(tmp_path):
    from endpoints import sovereignty as sovereignty_endpoints

    cache_dir = tmp_path / "storage_cache"
    cache_dir.mkdir()
    (cache_dir / "sample.cache").write_text("hello world")
    (cache_dir / "sample.meta").write_text('{"source":"test"}')

    agent = MagicMock(storage=MagicMock())
    app, original = _prepare_app(agent)
    original_cache_dir = sovereignty_endpoints.STORAGE_CACHE_DIR
    sovereignty_endpoints.STORAGE_CACHE_DIR = cache_dir
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                list_response = client.get("/api/sovereignty/files", headers={"X-API-Key": "test-key"})
                preview_response = client.get(
                    "/api/sovereignty/files/sample.cache/preview",
                    headers={"X-API-Key": "test-key"},
                )
                invalid_response = client.get(
                    "/api/sovereignty/files/..hidden/preview",
                    headers={"X-API-Key": "test-key"},
                )
        assert list_response.status_code == 200
        assert list_response.json()["file_count"] == 2
        assert preview_response.status_code == 200
        assert preview_response.json()["content"] == "hello world"
        assert invalid_response.status_code == 400
    finally:
        sovereignty_endpoints.STORAGE_CACHE_DIR = original_cache_dir
        _restore_app(app, original)
