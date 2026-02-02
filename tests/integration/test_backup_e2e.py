import pytest
import pytest_asyncio
import asyncio
import shutil
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root FIRST (before importing server)
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

from server import app, get_api_key
from kestrel_sovereign.inception_service import create_kestrel_identity_async
import kestrel_sovereign.storage as storage_pkg
import tempfile
import os


@pytest.fixture(scope="function")
def api_key():
    """Get the API key from environment."""
    return get_api_key()


@pytest_asyncio.fixture(scope="function")
async def async_client(monkeypatch):
    """Create an async client for testing the server with proper lifespan management."""
    # Use mkdtemp instead of TemporaryDirectory context manager to control cleanup
    agent_dir = tempfile.mkdtemp()
    try:
        # Set environment variable for the server to find the database
        monkeypatch.setenv("KESTREL_DB_PATH", agent_dir)
        # Set encryption key for backup tests
        monkeypatch.setenv("KESTREL_DATA_KEY", "test-key-for-backup-encryption")

        # Ensure server.Storage uses our temp directory
        monkeypatch.setattr(storage_pkg, "get_default_agent_data_dir", lambda: agent_dir)

        # Create agent identity in that directory (use async version since we're in async context)
        _ = await create_kestrel_identity_async(agent_dir)

        # Use LifespanManager to properly handle app startup/shutdown
        async with LifespanManager(app) as manager:
            transport = ASGITransport(app=manager.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                yield client
    finally:
        # Use shutil.rmtree with ignore_errors to handle files still in use
        shutil.rmtree(agent_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_backup_local_tier_via_api(async_client: AsyncClient, api_key: str):
    headers = {"X-API-Key": api_key}

    # health (doesn't require auth)
    resp = await async_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["agent_initialized"] is True

    # create local backup
    resp = await async_client.post("/agent/invoke", json={"input": "!backup tier=local"}, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "Backup created:" in data["response"]

    # verify a backup_artifact exists in graph via the agent's storage
    agent = app.state.agent
    backups = await agent.storage.get_nodes_by_type("backup_artifact")
    assert len(backups) >= 1


@pytest.mark.asyncio
async def test_promote_backup_isolated_flow(async_client: AsyncClient, api_key: str):
    headers = {"X-API-Key": api_key}

    # switch to isolated mode
    resp = await async_client.post("/agent/invoke", json={"input": "!privacy isolated"}, headers=headers)
    assert resp.status_code == 200

    # add activity in isolated session (use command that doesn't require LLM)
    resp = await async_client.post("/agent/invoke", json={"input": "!status"}, headers=headers)
    assert resp.status_code == 200

    # promote and back up
    resp = await async_client.post("/agent/invoke", json={"input": "!promote-backup tier=local"}, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "Backup created:" in data["response"]

    # verify a backup_artifact exists
    agent = app.state.agent
    backups = await agent.storage.get_nodes_by_type("backup_artifact")
    assert len(backups) >= 1
