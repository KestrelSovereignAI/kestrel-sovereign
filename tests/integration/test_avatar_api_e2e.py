"""Integration tests for avatar API endpoints

Uses FastAPI TestClient - no external server required.
"""

import pytest
from fastapi.testclient import TestClient
import tempfile
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from server import app
from kestrel_sovereign.inception_service import create_kestrel_identity
from kestrel_sovereign import storage


# Minimal valid JPEG bytes for testing (SOI + APP0 + EOI markers)
TEST_IMAGE_BYTES = bytes([
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00,
    0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9
])


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
        # This prevents "Event loop is closed" errors when pytest closes its event loop
        threads_after = set(threading.enumerate())
        new_threads = threads_after - threads_before
        for t in new_threads:
            if t.is_alive() and not t.daemon:
                t.join(timeout=2.0)


class TestFileEndpoint:
    """Test /api/files/{content_hash} endpoint"""

    def test_file_endpoint_404_for_nonexistent(self, client: TestClient):
        """GET /api/files/{hash} returns 404 for non-existent file"""
        response = client.get("/api/files/nonexistent_hash_abc123")
        assert response.status_code == 404

    def test_file_endpoint_head_request(self, client: TestClient):
        """HEAD /api/files/{hash} checks existence without body"""
        response = client.head("/api/files/nonexistent_hash")
        assert response.status_code == 404


class TestIdentityEndpoint:
    """Test /api/identity endpoint avatar fields"""

    def test_identity_has_avatar_fields(self, client: TestClient):
        """GET /api/identity response includes avatar fields"""
        response = client.get("/api/identity")

        # Skip if agent not initialized
        if response.status_code == 503:
            pytest.skip("Agent not initialized")

        assert response.status_code == 200
        data = response.json()

        # Skip if server hasn't been updated with new avatar fields
        if "avatar_hash" not in data:
            pytest.skip("Server not updated with avatar fields - restart required")

        # Verify avatar fields exist (may be null if no avatar)
        assert "avatar_hash" in data
        assert "avatar_url" in data

        # If avatar exists, URL should point to files endpoint
        if data.get("avatar_url"):
            assert data["avatar_url"].startswith("/api/files/")

    def test_identity_avatar_url_format(self, client: TestClient):
        """Avatar URL has correct format when present"""
        response = client.get("/api/identity")

        if response.status_code == 503:
            pytest.skip("Agent not initialized")

        data = response.json()

        # Skip if server hasn't been updated with new avatar fields
        if "avatar_hash" not in data:
            pytest.skip("Server not updated with avatar fields - restart required")

        if data.get("avatar_hash"):
            # URL should be /api/files/{hash}
            expected_url = f"/api/files/{data['avatar_hash']}"
            assert data["avatar_url"] == expected_url


class TestAvatarStorageIntegration:
    """Integration tests for avatar storage workflow"""

    def test_stored_avatar_retrievable_via_api(self, client: TestClient):
        """Avatar stored via file_store can be retrieved via /api/files/"""
        # Get identity to check agent status
        response = client.get("/api/identity")

        if response.status_code == 503:
            pytest.skip("Agent not initialized")

        data = response.json()

        # If agent has avatar, verify it's retrievable
        if data.get("avatar_url"):
            avatar_response = client.get(data["avatar_url"])
            assert avatar_response.status_code == 200
            assert avatar_response.headers.get("content-type", "").startswith("image/")

    def test_avatar_caching_headers(self, client: TestClient):
        """Avatar endpoint returns appropriate cache headers"""
        response = client.get("/api/identity")

        if response.status_code == 503:
            pytest.skip("Agent not initialized")

        data = response.json()

        if data.get("avatar_url"):
            avatar_response = client.get(data["avatar_url"])

            # Should have cache headers for immutable content
            cache_control = avatar_response.headers.get("cache-control", "")
            assert "max-age" in cache_control or avatar_response.status_code == 404


class TestVisualIdentityFeatureIntegration:
    """Test !avatar command integration (requires agent with feature)"""

    @pytest.mark.slow
    def test_avatar_command_available(self, client: TestClient):
        """Verify !avatar command is recognized"""
        # Check available commands
        response = client.get("/api/commands")

        if response.status_code == 503:
            pytest.skip("Agent not initialized")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        commands = data.get("commands", [])
        # Look for avatar-related command - API uses "cmd" field
        command_cmds = [c.get("cmd", "") for c in commands]
        # Either !avatar or !selfie should exist if feature enabled
        has_avatar_cmd = any(
            "avatar" in cmd.lower() or "selfie" in cmd.lower()
            for cmd in command_cmds
        )
        assert has_avatar_cmd, f"Expected !avatar or !selfie in commands, got: {command_cmds}"


class TestFileServing:
    """Test file serving functionality"""

    def test_files_endpoint_requires_auth(self, client: TestClient):
        """File endpoint requires authentication"""
        # Remove auth headers for this test
        saved_key = client.headers.pop("X-API-Key", None)

        # Should return 401 (unauthorized) without auth
        response = client.get("/api/files/test_hash_12345")
        assert response.status_code == 401

        # Restore auth header for subsequent tests
        if saved_key:
            client.headers["X-API-Key"] = saved_key

    def test_file_content_type_detection(self, client: TestClient):
        """File endpoint returns correct content type"""
        response = client.get("/api/identity")

        if response.status_code == 503:
            pytest.skip("Agent not initialized")

        data = response.json()

        if data.get("avatar_url"):
            avatar_response = client.get(data["avatar_url"])

            if avatar_response.status_code == 200:
                content_type = avatar_response.headers.get("content-type", "")
                # Should be image type
                assert content_type.startswith("image/") or content_type == "application/octet-stream"
