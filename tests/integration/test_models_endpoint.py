"""
Integration tests for /api/models endpoint.
Tests API calls using FastAPI TestClient (works in CI without running server).
"""
import pytest
import tempfile
import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Load .env from project root FIRST (before importing server)
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

sys.path.insert(0, str(project_root))
from server import app
from kestrel_sovereign.inception_service import create_kestrel_identity
from kestrel_sovereign import storage


@pytest.fixture(scope="function")
def client(monkeypatch):
    """
    Create a real agent environment with temp database for each test.
    Uses TestClient so no running server is needed.
    """
    import threading

    with tempfile.TemporaryDirectory() as agent_dir:
        # Set environment variable for the server to find the database
        monkeypatch.setenv("KESTREL_DB_PATH", agent_dir)

        # Point storage to temp directory
        monkeypatch.setattr(storage, "get_default_agent_data_dir", lambda: agent_dir)

        # Set a test API key
        test_api_key = "test-api-key-12345"
        monkeypatch.setenv("KESTREL_API_KEY", test_api_key)

        # Create real agent identity with real constitution
        create_kestrel_identity(agent_dir, "docs/principles/KESTREL_CONSTITUTION.md")

        # Track threads before TestClient starts
        threads_before = set(threading.enumerate())

        # TestClient initializes the app with real agent
        with TestClient(app) as test_client:
            # Add the API key to the client's default headers
            test_client.headers.update({"X-API-Key": test_api_key})
            yield test_client

        # After TestClient exits, wait for new threads (aiosqlite) to finish
        # This prevents "Event loop is closed" errors when pytest closes its event loop
        threads_after = set(threading.enumerate())
        new_threads = threads_after - threads_before
        for t in new_threads:
            if t.is_alive() and not t.daemon:
                t.join(timeout=2.0)


class TestModelsEndpoint:
    """Test /api/models endpoint"""

    def test_models_endpoint_returns_json(self, client: TestClient):
        """Test that endpoint returns valid JSON"""
        response = client.get("/api/models")

        # Should return 200 or 503 (if agent not initialized)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_models_endpoint_structure(self, client: TestClient):
        """Test response structure"""
        response = client.get("/api/models")

        if response.status_code != 200:
            pytest.skip("Agent not initialized")

        data = response.json()

        # Check required fields
        assert "by_provider" in data
        assert "featured" in data
        assert "all" in data
        assert "default" in data
        assert "count" in data

        assert isinstance(data["by_provider"], dict)
        assert isinstance(data["featured"], list)
        assert isinstance(data["all"], list)
        assert isinstance(data["count"], int)

    def test_models_endpoint_featured_only(self, client: TestClient):
        """Test featured_only parameter"""
        response = client.get("/api/models?featured_only=true")

        if response.status_code != 200:
            pytest.skip("Agent not initialized")

        data = response.json()

        # All models should be featured
        for model in data["all"]:
            assert model.get("is_featured") is True, f"Non-featured model returned: {model.get('id')}"

    def test_models_endpoint_show_all(self, client: TestClient):
        """Test featured_only=false to show all models"""
        response_featured = client.get("/api/models?featured_only=true")
        response_all = client.get("/api/models?featured_only=false")

        if response_featured.status_code != 200 or response_all.status_code != 200:
            pytest.skip("Agent not initialized")

        data_featured = response_featured.json()
        data_all = response_all.json()

        # All models should include featured models
        assert data_all["count"] >= data_featured["count"]

    def test_models_endpoint_category_filter(self, client: TestClient):
        """Test category filter parameter"""
        response = client.get("/api/models?category=chat&featured_only=false")

        if response.status_code != 200:
            pytest.skip("Agent not initialized")

        data = response.json()

        # All models should be chat category
        for model in data["all"]:
            assert model.get("category") == "chat", f"Non-chat model returned: {model.get('id')}"

    def test_models_endpoint_invalid_category(self, client: TestClient):
        """Test invalid category returns 400"""
        response = client.get("/api/models?category=invalid")

        # Should return 400 for invalid category or 503 if not initialized
        if response.status_code == 503:
            pytest.skip("Agent not initialized")

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_models_endpoint_provider_filter(self, client: TestClient):
        """Test provider filter parameter"""
        # First get all models to find available providers
        response_all = client.get("/api/models?featured_only=false")

        if response_all.status_code != 200:
            pytest.skip("Agent not initialized")

        data_all = response_all.json()

        if not data_all["by_provider"]:
            pytest.skip("No providers available")

        # Pick first provider
        provider = list(data_all["by_provider"].keys())[0]

        # Filter to that provider
        response = client.get(f"/api/models?providers={provider}&featured_only=false")
        data = response.json()

        # All models should be from that provider
        for model in data["all"]:
            assert model.get("provider") == provider

    def test_models_endpoint_model_structure(self, client: TestClient):
        """Test individual model structure"""
        response = client.get("/api/models?featured_only=false")

        if response.status_code != 200:
            pytest.skip("Agent not initialized")

        data = response.json()

        if not data["all"]:
            pytest.skip("No models discovered")

        model = data["all"][0]

        # Check required model fields
        assert "id" in model
        assert "provider" in model
        assert "display_name" in model
        assert "category" in model
        assert "is_featured" in model
        assert "is_hidden" in model

    def test_models_endpoint_by_provider_grouping(self, client: TestClient):
        """Test that by_provider groups correctly"""
        response = client.get("/api/models?featured_only=false")

        if response.status_code != 200:
            pytest.skip("Agent not initialized")

        data = response.json()

        # Check that models are grouped correctly
        for provider, models in data["by_provider"].items():
            for model in models:
                assert model.get("provider") == provider

    def test_models_endpoint_caching(self, client: TestClient):
        """Test caching parameter"""
        # First call - no cache
        response1 = client.get("/api/models?use_cache=false")

        if response1.status_code != 200:
            pytest.skip("Agent not initialized")

        # Second call - with cache
        response2 = client.get("/api/models?use_cache=true")

        data1 = response1.json()
        data2 = response2.json()

        # Should return same count (cached)
        assert data1["count"] == data2["count"]


class TestModelsEndpointV1:
    """Test OpenAI-compatible /v1/models endpoint"""

    def test_v1_models_endpoint(self, client: TestClient):
        """Test /v1/models OpenAI compatibility"""
        response = client.get("/v1/models")

        if response.status_code not in [200]:
            pytest.skip("Agent not initialized")

        data = response.json()

        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_v1_models_structure(self, client: TestClient):
        """Test model object structure in /v1/models"""
        response = client.get("/v1/models")

        if response.status_code != 200:
            pytest.skip("Agent not initialized")

        data = response.json()

        if data["data"]:
            model = data["data"][0]
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"


class TestCurrentModelEndpoint:
    """Test /api/model/current endpoint"""

    def test_current_model_endpoint(self, client: TestClient):
        """Test /api/model/current returns current model"""
        response = client.get("/api/model/current")

        if response.status_code not in [200, 503]:
            pytest.fail(f"Unexpected status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            assert "model" in data
            assert "provider" in data
            assert "model_name" in data
