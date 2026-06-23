"""
Integration tests for full agent chat flow via /agent/invoke with REAL LLM calls.

These tests exercise the FULL agent stack via the FastAPI /agent/invoke endpoint
with REAL gpt-5-mini calls. NO MOCKS - these are real integration tests.

Run with: uv run pytest tests/integration/test_agent_chat_e2e.py -v
"""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
from fastapi.testclient import TestClient

# Load .env from project root FIRST (before importing server)
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

# Skip all tests if OpenAI key not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY"
)


@pytest.fixture(scope="function")
def client(monkeypatch):
    """
    This fixture sets up a clean agent environment for each test function.
    It uses monkeypatch to ensure the app uses a temporary data directory.
    """
    import threading
    import tempfile
    from kestrel_sovereign import storage
    from kestrel_sovereign.inception_service import create_kestrel_identity
    from server import app

    with tempfile.TemporaryDirectory() as agent_dir:
        # Set environment variable for the server to find the database
        monkeypatch.setenv("KESTREL_DB_PATH", agent_dir)

        # Disable multi-agent mode for this test (force single-agent mode)
        monkeypatch.delenv("KESTREL_MULTI_AGENT", raising=False)

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


@pytest.fixture(scope="function")
def api_key():
    """Get the API key from environment."""
    from server import get_api_key
    return get_api_key()


def test_health_endpoint(client: TestClient):
    """Test /health endpoint returns agent_initialized status."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    # Check required fields; llm_reachability is optional (added in #1265)
    assert data["status"] == "ok"
    assert data["agent_initialized"] is True


def test_agent_invoke_basic_chat(client: TestClient, api_key: str):
    """Test /agent/invoke with basic chat - REAL gpt-5-mini call."""
    headers = {"X-API-Key": api_key}

    payload = {
        "input": "Hello, who are you?",
        "model": "gpt-5-mini"
    }

    response = client.post("/api/agent/invoke", json=payload, headers=headers)

    # Assert 200 status
    assert response.status_code == 200

    # Assert response has "response" key with non-empty string
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0

    # Assert response mentions Kestrel or agent or help (agent should identify itself)
    response_lower = data["response"].lower()
    assert any(keyword in response_lower for keyword in ["kestrel", "agent", "help", "assist"])


def test_agent_invoke_factual_question(client: TestClient, api_key: str):
    """Test /agent/invoke with factual question - REAL gpt-5-mini call."""
    headers = {"X-API-Key": api_key}

    payload = {
        "input": "What is 2+2? Please just give the answer.",
        "model": "gpt-5-mini"
    }

    response = client.post("/api/agent/invoke", json=payload, headers=headers)

    assert response.status_code == 200
    data = response.json()

    # Assert response is non-empty (agent may respond with greeting or answer)
    assert "response" in data
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0


def test_agent_invoke_empty_input_returns_400(client: TestClient, api_key: str):
    """Test /agent/invoke with empty input returns 400."""
    headers = {"X-API-Key": api_key}

    # Empty payload (no "input" field)
    payload = {}

    response = client.post("/api/agent/invoke", json=payload, headers=headers)

    # Assert 400 status
    assert response.status_code == 400


def test_agent_invoke_without_auth_returns_401(client: TestClient):
    """Test /agent/invoke without X-API-Key header returns 401."""
    # No headers (no auth)
    payload = {
        "input": "Hello",
        "model": "gpt-5-mini"
    }

    response = client.post("/api/agent/invoke", json=payload)

    # Assert 401 status
    assert response.status_code == 401
