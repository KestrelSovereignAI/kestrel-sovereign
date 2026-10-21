"""Original native read/commit controls at public constitutional/API paths."""
import asyncio
from contextlib import asynccontextmanager

import aiosqlite
import httpx
import pytest
from fastapi import FastAPI

from kestrel_sovereign.api_errors import register_api_error_handlers
from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_constitution_turn_admission import _ready_turn
from tests.unit.test_execution_custody_review43_native import carries


@asynccontextmanager
async def native_read_storage(path, carrier):
    """Physically retire the read owner and verify its retained control debt."""
    storage = AsyncStorage(str(path), agent_id="did:test:governing55")
    await storage.initialize()
    try:
        yield storage
    finally:
        try:
            await storage.close()
        except asyncio.CancelledError as error:
            # The native cancelled-read owner retains the original carrier
            # through its drain. Close must report it AFTER worker retirement.
            assert carrier == "cancel"
            assert carries(error, storage._review55_original_read_error)
        assert storage.db.backend._connection is None


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["error", "cancel", "group"])
async def test_native_governing_identity_read_controls_are_not_error_text(tmp_path, monkeypatch, carrier):
    async with native_read_storage(tmp_path / "governing.db", carrier) as storage:
        agent, _ = await _ready_turn(storage)
        native_execute = aiosqlite.Connection._execute
        connection = storage.db.backend._connection
        original = asyncio.CancelledError("native governing read ACK") if carrier == "cancel" else OSError("native governing read ACK")
        original.__cause__ = ExecutionCommitOutcomeError("unknown")
        if carrier == "group":
            original = BaseExceptionGroup("native identity read", [ValueError("ordinary"), original])
        storage._review55_original_read_error = original
        acknowledgements = []
        async def lost_ack(target, function, *args, **kwargs):
            result = await native_execute(target, function, *args, **kwargs)
            if target is connection and args and isinstance(args[0], str) and args[0].lstrip().startswith("SELECT") and "graph_nodes" in args[0] and not acknowledgements:
                acknowledgements.append(True)
                raise original
            return result
        monkeypatch.setattr(aiosqlite.Connection, "_execute", lost_ack)
        try:
            with pytest.raises(BaseException) as caught:
                await agent._get_governing_constitution()
            assert acknowledgements == [True]
            assert carries(caught.value, original)
        finally:
            monkeypatch.undo()


@pytest.mark.asyncio
async def test_public_api_classifies_actual_commit_ack_as_terminal_control(tmp_path, monkeypatch):
    async with AsyncStorage(str(tmp_path / "wire.db"), agent_id="did:test:wire55") as storage:
        await storage.db.execute_commit("CREATE TABLE effects(id INTEGER)")
        connection = storage.db.backend._connection
        native_commit = aiosqlite.Connection.commit
        original = OSError("private native commit detail must not be emitted")
        original.__cause__ = ExecutionCommitOutcomeError("unknown")
        async def lost_ack(target):
            await native_commit(target)
            if target is connection:
                raise original
        monkeypatch.setattr(aiosqlite.Connection, "commit", lost_ack)
        app = FastAPI()
        register_api_error_handlers(app)
        @app.post("/effect")
        async def effect():
            await storage.db.execute_commit("INSERT INTO effects VALUES (1)")
            return {"result": "unreachable"}
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app, raise_app_exceptions=False), base_url="http://offline.invalid") as client:
                response = await client.post("/effect")
            assert response.status_code == 500
            assert response.json()["error"]["code"] == "execution_control"
            assert "private native commit detail" not in response.text
            assert await storage.db.fetchval("SELECT COUNT(*) FROM effects") == 1
        finally:
            monkeypatch.undo()
