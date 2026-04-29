"""
End-to-End API Test

Tests the Kestrel agent's FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root FIRST (before importing server)
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

sys.path.insert(0, str(project_root))
from server import app, get_api_key
from kestrel_sovereign.inception_service import create_kestrel_identity
from kestrel_sovereign import storage
import shutil
import tempfile


@pytest.fixture(scope="function")
def api_key():
    """Get the API key from environment."""
    return get_api_key()


@pytest.fixture(scope="function")
def client(monkeypatch):
    """
    This fixture sets up a clean agent environment for each test function.
    It uses monkeypatch to ensure the app uses a temporary data directory.
    """
    import threading

    with tempfile.TemporaryDirectory() as agent_dir:
        # Set environment variable for the server to find the database
        monkeypatch.setenv("KESTREL_DB_PATH", agent_dir)

        # Monkeypatch the function that returns the default data directory
        monkeypatch.setattr(storage, "get_default_agent_data_dir", lambda: agent_dir)

        # Now, create the identity in that directory
        create_kestrel_identity(agent_dir, "docs/principles/KESTREL_CONSTITUTION.md")

        # Track threads before TestClient starts
        threads_before = set(threading.enumerate())

        # The TestClient will now initialize the app, which will
        # in turn initialize Storage using the patched directory.
        with TestClient(app) as client:
            yield client

        # After TestClient exits, wait for new threads (aiosqlite) to finish
        threads_after = set(threading.enumerate())
        new_threads = threads_after - threads_before
        for t in new_threads:
            if t.is_alive() and not t.daemon:
                t.join(timeout=2.0)

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", "").strip() and not os.environ.get("ANTHROPIC_API_KEY", "").strip(),
    reason="Requires LLM API key to invoke agent"
)
def test_invoke_agent_e2e(client: TestClient, api_key: str):
    """
    End-to-end test for invoking the agent.
    It simulates a user sending a message and getting a response.
    """
    headers = {"X-API-Key": api_key}

    # 1. Check health before invoking (health endpoint doesn't require auth)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "agent_initialized": True}

    # 2. Invoke the agent with API key
    payload = {"input": "Hello, who are you?"}
    response = client.post("/api/agent/invoke", json=payload, headers=headers)

    # 3. Check the response
    assert response.status_code == 200, f"Failed with {response.status_code}: {response.text}"
    response_data = response.json()
    assert "response" in response_data
    assert isinstance(response_data["response"], str)
    # When LLM is not configured, sovereignty feature provides constitutional correction
    assert ("Kestrel" in response_data["response"] or
            "constitution" in response_data["response"].lower() or
            "system_correction" in response_data["response"].lower() or
            "can't help" in response_data["response"].lower() or
            "agent" in response_data["response"].lower())
