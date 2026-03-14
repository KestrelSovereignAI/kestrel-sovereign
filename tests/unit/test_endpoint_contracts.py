import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from endpoints.commands import router as commands_router, BUILTIN_COMMANDS
from endpoints.files import router as files_router
from endpoints.observability import router as observability_router
from endpoints.database import _get_table_columns, _list_table_names


class TestCommandsEndpoint:
    def test_commands_endpoint_keeps_object_shape_without_agent(self):
        app = FastAPI()
        app.include_router(commands_router)

        with TestClient(app) as client:
            response = client.get("/api/commands")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert isinstance(data["commands"], list)
        assert data["count"] == len(BUILTIN_COMMANDS)


class TestObservabilityEndpoints:
    def test_observability_events_returns_503_without_agent(self):
        app = FastAPI()
        app.include_router(observability_router)

        with TestClient(app) as client:
            response = client.get("/api/observability/events")

        assert response.status_code == 503
        assert "agent" in response.json()["detail"].lower()

    def test_observability_events_returns_503_without_store(self):
        app = FastAPI()
        app.include_router(observability_router)
        app.state.agent = MagicMock(observability_store=None)

        with TestClient(app) as client:
            response = client.get("/api/observability/events")

        assert response.status_code == 503
        assert "observability store" in response.json()["detail"].lower()

    def test_observability_summary_returns_500_on_store_failure(self):
        app = FastAPI()
        app.include_router(observability_router)
        app.state.agent = MagicMock(
            observability_store=MagicMock(query_events=AsyncMock(side_effect=RuntimeError("boom")))
        )

        with TestClient(app) as client:
            response = client.get("/api/observability/summary")

        assert response.status_code == 500
        assert "observability" in response.json()["detail"].lower()


class TestFilesEndpoint:
    def test_head_uses_file_existence_not_metadata_presence(self):
        file_store = MagicMock()
        file_store.file_exists = AsyncMock(return_value=True)
        file_store.get_file_metadata = AsyncMock(return_value=None)

        agent = MagicMock(storage=MagicMock(files=file_store))
        app = FastAPI()
        app.include_router(files_router)
        app.state.agent = agent

        with TestClient(app) as client:
            response = client.head("/api/files/test-hash")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/octet-stream")


class TestDatabaseExplorerHelpers:
    @pytest.mark.asyncio
    async def test_list_table_names_uses_postgres_catalog(self):
        db = MagicMock()
        db.backend_type = "postgres"
        db.fetchall = AsyncMock(return_value=[("saved_items",), ("conversation_history",)])

        table_names = await _list_table_names(db)

        assert table_names == ["saved_items", "conversation_history"]
        query = db.fetchall.await_args.args[0]
        assert "information_schema.tables" in query

    @pytest.mark.asyncio
    async def test_get_table_columns_uses_postgres_information_schema(self):
        db = MagicMock()
        db.backend_type = "postgres"
        db.fetchall = AsyncMock(return_value=[
            ("id", "uuid", "NO", True),
            ("name", "text", "YES", False),
        ])

        columns = await _get_table_columns(db, "saved_items")

        assert columns == [
            {"name": "id", "type": "uuid", "nullable": False, "pk": True},
            {"name": "name", "type": "text", "nullable": True, "pk": False},
        ]
        query = db.fetchall.await_args.args[0]
        assert "information_schema.columns" in query
