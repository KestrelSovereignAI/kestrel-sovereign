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
        # Prove local backup tests are isolated from ambient cloud sync credentials.
        monkeypatch.setenv("LIGHTHOUSE_API_KEY", "test-lighthouse-key")
        monkeypatch.setenv("GCS_BACKUP_BUCKET", "test-backup-bucket")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(Path(agent_dir) / "missing-gcs-creds.json"))
        monkeypatch.setenv("KESTREL_SYNC_ENABLED", "false")

        # Ensure server.Storage uses our temp directory
        monkeypatch.setattr(storage_pkg, "get_default_agent_data_dir", lambda: agent_dir)

        # Create agent identity in that directory (use async version since we're in async context)
        _ = await create_kestrel_identity_async(agent_dir)

        # Use LifespanManager to properly handle app startup/shutdown
        async with LifespanManager(app) as manager:
            # Skip bootstrap for test agents
            from tests.integration.conftest import complete_bootstrap
            await complete_bootstrap(app.state.agent)

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
async def test_backup_e2e_disables_ambient_sync_targets(async_client: AsyncClient):
    agent = app.state.agent

    assert os.environ["LIGHTHOUSE_API_KEY"] == "test-lighthouse-key"
    assert os.environ["GCS_BACKUP_BUCKET"] == "test-backup-bucket"
    assert getattr(agent, "_sync_service", None) is None


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
