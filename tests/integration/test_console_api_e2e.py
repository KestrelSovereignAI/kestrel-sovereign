"""
End-to-End Integration Tests for Kestrel Sovereign Console API

NO MOCKS - Tests use real storage, real agent initialization.
Uses FastAPI TestClient with temporary database.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from server import app
from kestrel_sovereign.inception_service import create_kestrel_identity
from kestrel_sovereign import storage


@pytest.fixture(scope="function")
def client(monkeypatch):
    """
    Create a real agent environment with temp database for each test.
    NO MOCKS - real storage, real inception.
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
        with TestClient(app) as client:
            # Add the API key to the client's default headers
            client.headers.update({"X-API-Key": test_api_key})
            yield client

        # After TestClient exits, wait for new threads (aiosqlite) to finish
        threads_after = set(threading.enumerate())
        new_threads = threads_after - threads_before
        for t in new_threads:
            if t.is_alive() and not t.daemon:
                t.join(timeout=2.0)


class TestHealthEndpoint:
    """Health check must work before any other tests."""

    def test_health_returns_ok(self, client: TestClient):
        """Health endpoint returns status ok with agent initialized."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["agent_initialized"] is True


class TestIdentityAPI:
    """Tests for /api/identity endpoint."""

    def test_get_identity_returns_did(self, client: TestClient):
        """Identity endpoint returns valid DID."""
        response = client.get("/api/identity")
        assert response.status_code == 200
        data = response.json()

        # Must have a DID
        assert "did" in data
        assert data["did"] is not None
        assert data["did"].startswith("did:")

    def test_get_identity_returns_constitution_hash(self, client: TestClient):
        """Identity endpoint returns constitution hash."""
        response = client.get("/api/identity")
        assert response.status_code == 200
        data = response.json()

        # Must have constitution hash
        assert "constitution_hash" in data
        assert data["constitution_hash"] is not None
        assert len(data["constitution_hash"]) == 64  # SHA256 hex

    def test_get_identity_returns_node_type(self, client: TestClient):
        """Identity endpoint returns node_type as 'agent'."""
        response = client.get("/api/identity")
        assert response.status_code == 200
        data = response.json()

        assert "node_type" in data
        assert data["node_type"] == "agent"

    def test_get_identity_has_created_at(self, client: TestClient):
        """Identity endpoint returns created_at timestamp."""
        response = client.get("/api/identity")
        assert response.status_code == 200
        data = response.json()

        assert "created_at" in data
        # created_at should be a valid ISO timestamp or None
        if data["created_at"]:
            assert "T" in data["created_at"] or "-" in data["created_at"]


class TestConstitutionAPI:
    """Tests for /api/constitution endpoint."""

    def test_get_constitution_returns_text(self, client: TestClient):
        """Constitution endpoint returns constitution text."""
        response = client.get("/api/constitution")
        assert response.status_code == 200
        data = response.json()

        assert "text" in data
        assert data["text"] is not None
        assert len(data["text"]) > 100  # Constitution should have content

    def test_get_constitution_returns_hash(self, client: TestClient):
        """Constitution endpoint returns matching hash in metadata."""
        response = client.get("/api/constitution")
        assert response.status_code == 200
        data = response.json()

        assert "metadata" in data
        assert "hash" in data["metadata"]
        assert data["metadata"]["hash"] is not None
        assert len(data["metadata"]["hash"]) == 64  # SHA256 hex

    def test_constitution_hash_matches_identity(self, client: TestClient):
        """Constitution hash matches the one in identity."""
        identity_response = client.get("/api/identity")
        constitution_response = client.get("/api/constitution")

        identity_hash = identity_response.json().get("constitution_hash")
        constitution_hash = constitution_response.json().get("metadata", {}).get("hash")

        assert identity_hash == constitution_hash

    def test_constitution_contains_kestrel_principles(self, client: TestClient):
        """Constitution text contains Kestrel principles."""
        response = client.get("/api/constitution")
        data = response.json()
        text = data.get("text", "").lower()

        # Should mention sovereignty or constitution concepts
        assert "sovereign" in text or "constitution" in text or "kestrel" in text


class TestSessionsAPI:
    """Tests for /api/sessions endpoint."""

    def test_get_sessions_returns_list(self, client: TestClient):
        """Sessions endpoint returns messages list."""
        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()

        assert "messages" in data
        assert isinstance(data["messages"], list)

    def test_get_sessions_returns_counts(self, client: TestClient):
        """Sessions endpoint returns message counts."""
        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()

        assert "total" in data
        assert "user_messages" in data
        assert "agent_messages" in data
        assert isinstance(data["total"], int)

    def test_get_sessions_respects_limit(self, client: TestClient):
        """Sessions endpoint respects limit parameter."""
        response = client.get("/api/sessions?limit=5")
        assert response.status_code == 200
        data = response.json()

        # Even if empty, should not exceed limit
        assert len(data["messages"]) <= 5


class TestMemoriesAPI:
    """Tests for /api/memories endpoint."""

    def test_get_memories_returns_nodes(self, client: TestClient):
        """Memories endpoint returns knowledge graph nodes."""
        response = client.get("/api/memories")
        assert response.status_code == 200
        data = response.json()

        assert "nodes" in data
        assert isinstance(data["nodes"], list)
        assert "total" in data

    def test_get_memories_includes_agent_node(self, client: TestClient):
        """Memories should include the agent node itself."""
        response = client.get("/api/memories")
        data = response.json()

        nodes = data.get("nodes", [])
        node_types = [n.get("node_type") for n in nodes]

        # Agent node should exist
        assert "agent" in node_types

    def test_get_memories_filter_by_type(self, client: TestClient):
        """Memories endpoint filters by node_type."""
        response = client.get("/api/memories?node_type=agent")
        assert response.status_code == 200
        data = response.json()

        for node in data.get("nodes", []):
            assert node.get("node_type") == "agent"

    def test_delete_memory_protects_agent_node(self, client: TestClient):
        """Cannot delete the agent node itself."""
        # First get the agent node
        response = client.get("/api/memories?node_type=agent")
        nodes = response.json().get("nodes", [])

        if nodes:
            agent_node_id = nodes[0].get("node_id")

            # Try to delete it
            delete_response = client.delete(f"/api/memories/{agent_node_id}")

            # Should be forbidden
            assert delete_response.status_code == 403
            assert "agent" in delete_response.json().get("detail", "").lower()


class TestStorageStatsAPI:
    """Tests for /api/storage/stats endpoint."""

    def test_get_storage_stats_returns_database_info(self, client: TestClient):
        """Storage stats returns database information."""
        response = client.get("/api/storage/stats")
        assert response.status_code == 200
        data = response.json()

        assert "database" in data
        assert "path" in data["database"]
        assert "size_bytes" in data["database"]
        assert isinstance(data["database"]["size_bytes"], int)

    def test_get_storage_stats_returns_counts(self, client: TestClient):
        """Storage stats returns conversation and node counts."""
        response = client.get("/api/storage/stats")
        assert response.status_code == 200
        data = response.json()

        assert "conversations" in data
        assert "graph_nodes" in data
        # conversations is an object with count, graph_nodes is a dict of types
        assert isinstance(data["conversations"], dict)
        assert isinstance(data["graph_nodes"], dict)

    def test_get_storage_stats_returns_sovereignty_info(self, client: TestClient):
        """Storage stats returns sovereignty export count."""
        response = client.get("/api/storage/stats")
        assert response.status_code == 200
        data = response.json()

        assert "sovereignty_exports" in data
        assert isinstance(data["sovereignty_exports"], int)


class TestSovereigntyAPI:
    """Tests for /api/sovereignty/* endpoints."""

    def test_get_exports_returns_list(self, client: TestClient):
        """Sovereignty exports endpoint returns list."""
        response = client.get("/api/sovereignty/exports")
        assert response.status_code == 200
        data = response.json()

        assert "exports" in data
        assert isinstance(data["exports"], list)

    def test_export_creates_receipt(self, client: TestClient):
        """POST to sovereignty export creates a receipt."""
        # Trigger export
        response = client.post(
            "/api/sovereignty/export",
            json={"tier": "local", "encrypt": True}
        )

        # Should succeed (may fallback to local cache)
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True or "message" in data

    def test_import_requires_cid(self, client: TestClient):
        """Import endpoint requires a CID."""
        response = client.post(
            "/api/sovereignty/import",
            json={}
        )

        # Should fail with validation error or 400
        assert response.status_code in [400, 422]


class TestWalletAPI:
    """Tests for /api/wallet endpoint."""

    def test_get_wallet_returns_balance(self, client: TestClient):
        """Wallet endpoint returns balance information."""
        response = client.get("/api/wallet")
        assert response.status_code == 200
        data = response.json()

        assert "balance" in data
        assert "currency" in data

    def test_wallet_has_audit_reserve(self, client: TestClient):
        """Wallet includes audit reserve information."""
        response = client.get("/api/wallet")
        assert response.status_code == 200
        data = response.json()

        assert "audit_reserve" in data
        assert "total" in data


class TestModelsAPI:
    """Tests for /api/models endpoint."""

    def test_get_models_returns_list(self, client: TestClient):
        """Models endpoint returns list of available models."""
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()

        # API returns "all" key with list of models
        assert "all" in data
        assert isinstance(data["all"], list)

    def test_models_have_required_fields(self, client: TestClient):
        """Each model has id and provider fields."""
        response = client.get("/api/models")
        data = response.json()

        for model in data.get("all", []):
            assert "id" in model
            assert "provider" in model


class TestAgentInvokeWithConsole:
    """Tests for /agent/invoke endpoint used by console chat."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", "").strip() and not os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        reason="Requires LLM API key to invoke agent"
    )
    def test_invoke_returns_response(self, client: TestClient):
        """Agent invoke returns a response."""
        response = client.post(
            "/api/agent/invoke",
            json={"input": "What is your DID?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert isinstance(data["response"], str)

    def test_invoke_with_privacy_command(self, client: TestClient):
        """Agent handles !commands for privacy."""
        response = client.post(
            "/api/agent/invoke",
            json={"input": "!get-privacy-mode"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data


class TestPrivacyModeAPI:
    """Tests for privacy mode endpoints."""

    def test_get_privacy_mode(self, client: TestClient):
        """Get current privacy mode."""
        response = client.get("/api/agent/privacy-mode")
        assert response.status_code == 200
        data = response.json()

        assert "privacy_mode" in data
        # Default should be normal (lowercase from API)
        assert data["privacy_mode"].upper() in ["EPHEMERAL", "ISOLATED", "ANONYMOUS", "NORMAL", "PUBLIC"]

    def test_set_privacy_mode(self, client: TestClient):
        """Set privacy mode and verify change."""
        # Set to ISOLATED
        response = client.post(
            "/api/agent/privacy-mode",
            headers={"X-Kestrel-Allow-Destructive": "test-rail-bypass"},
json={"mode": "ISOLATED"}
        )
        assert response.status_code == 200

        # Verify it changed
        get_response = client.get("/api/agent/privacy-mode")
        assert get_response.json()["privacy_mode"].upper() == "ISOLATED"

    def test_set_invalid_privacy_mode_fails(self, client: TestClient):
        """Setting invalid privacy mode returns error."""
        response = client.post(
            "/api/agent/privacy-mode",
            headers={"X-Kestrel-Allow-Destructive": "test-rail-bypass"},
json={"mode": "INVALID_MODE"}
        )

        # Should fail
        assert response.status_code in [400, 422]
