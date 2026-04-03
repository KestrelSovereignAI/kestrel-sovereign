"""
Integration tests for model mandate command and API.

Tests the !model-mandate command and /api/models endpoint.
Uses TestClient for API tests - no running server required.
"""
from contextlib import asynccontextmanager
import pytest
from typing import Dict, Any
from fastapi.testclient import TestClient


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


class TestModelMandateMethods:
    """Tests for LLMService model mandate methods (no server needed)."""

    @pytest.fixture
    def llm_service(self):
        """Create an LLMService instance."""
        from kestrel_sovereign.llm.service import LLMService
        return LLMService()

    def test_get_current_mandate_returns_structure(self, llm_service):
        """get_current_mandate() returns expected structure."""
        mandate = llm_service.get_current_mandate()

        assert "preference" in mandate
        assert "model" in mandate["preference"]
        assert "provider" in mandate["preference"]
        assert "fallbacks" in mandate
        assert "banned" in mandate
        assert "mandates" in mandate
        assert isinstance(mandate["fallbacks"], list)
        assert isinstance(mandate["banned"], list)
        assert isinstance(mandate["mandates"], dict)

    def test_get_current_mandate_loads_toml_defaults(self, llm_service):
        """get_current_mandate() loads defaults from model_mandate.toml."""
        mandate = llm_service.get_current_mandate()

        # Should have a default preferred model from TOML
        # The exact value depends on model_mandate.toml content
        # Just verify the structure is populated
        assert mandate["preference"] is not None

    def test_set_model_preference(self, llm_service):
        """set_model_preference() changes preference."""
        llm_service.set_model_preference("test-model-123", "test-provider")

        mandate = llm_service.get_current_mandate()
        assert mandate["preference"]["model"] == "test-model-123"
        assert mandate["preference"]["provider"] == "test-provider"

    def test_set_model_preference_without_provider(self, llm_service):
        """set_model_preference() works without provider."""
        llm_service.set_model_preference("solo-model")

        mandate = llm_service.get_current_mandate()
        assert mandate["preference"]["model"] == "solo-model"
        assert mandate["preference"]["provider"] is None

    def test_add_fallback_model(self, llm_service):
        """add_fallback_model() adds to fallback list."""
        llm_service.add_fallback_model("fallback-1", "provider-a")
        llm_service.add_fallback_model("fallback-2", "provider-b")

        mandate = llm_service.get_current_mandate()
        assert len(mandate["fallbacks"]) == 2
        assert {"model": "fallback-1", "provider": "provider-a"} in mandate["fallbacks"]
        assert {"model": "fallback-2", "provider": "provider-b"} in mandate["fallbacks"]

    def test_add_fallback_model_no_duplicates(self, llm_service):
        """add_fallback_model() prevents duplicates."""
        llm_service.add_fallback_model("same-model", "same-provider")
        llm_service.add_fallback_model("same-model", "same-provider")

        mandate = llm_service.get_current_mandate()
        assert len(mandate["fallbacks"]) == 1

    def test_clear_mandate_resets_preference(self, llm_service):
        """clear_mandate() resets to TOML defaults."""
        # Set custom preference
        llm_service.set_model_preference("custom-model")
        llm_service.add_fallback_model("custom-fallback")

        # Verify it was set
        mandate = llm_service.get_current_mandate()
        assert mandate["preference"]["model"] == "custom-model"
        assert len(mandate["fallbacks"]) == 1

        # Clear it
        llm_service.clear_mandate()

        # Verify it was cleared
        mandate = llm_service.get_current_mandate()
        assert mandate["preference"]["model"] != "custom-model"
        assert len(mandate["fallbacks"]) == 0


class TestModelDiscoveryAPI:
    """Tests for /api/models endpoint using TestClient."""

    @pytest.fixture
    def client(self, monkeypatch):
        """TestClient with mock agent for API testing."""
        from server import app
        from unittest.mock import MagicMock
        from kestrel_sovereign.llm.service import LLMService

        test_api_key = "test-api-key-12345"
        monkeypatch.setenv("KESTREL_API_KEY", test_api_key)

        # Create mock agent with real LLMService
        mock_agent = MagicMock()
        mock_agent.agent_id = "did:test:model_mandate"
        mock_agent.llm_service = LLMService()
        mock_agent.storage = None

        # Mock discover_all_models to return sample data
        async def mock_discover(*args, **kwargs):
            from kestrel_sovereign.llm.model_metadata import ModelInfo
            return [
                ModelInfo(id="gpt-4", provider="openai", display_name="GPT-4", is_featured=True),
                ModelInfo(id="llama3.2:3b", provider="ollama", display_name="Llama 3.2"),
            ]

        mock_agent.llm_service.discover_all_models = mock_discover

        original_lifespan = app.router.lifespan_context
        app.router.lifespan_context = _noop_lifespan
        app.state.agent = mock_agent
        try:
            with TestClient(app) as client:
                client.headers.update({"X-API-Key": test_api_key})
                yield client
        finally:
            app.router.lifespan_context = original_lifespan
            app.state.agent = None

    def test_api_models_returns_by_provider(self, client):
        """GET /api/models returns by_provider grouped format."""
        response = client.get("/api/models")

        assert response.status_code == 200
        data = response.json()
        assert "by_provider" in data
        assert isinstance(data["by_provider"], dict)

    def test_api_models_returns_all_list(self, client):
        """GET /api/models returns all models list."""
        response = client.get("/api/models")

        data = response.json()
        assert "all" in data
        assert isinstance(data["all"], list)

    def test_api_models_has_default(self, client):
        """GET /api/models includes default model."""
        response = client.get("/api/models")

        data = response.json()
        assert "default" in data

    def test_api_models_has_featured(self, client):
        """GET /api/models returns featured models."""
        response = client.get("/api/models")

        data = response.json()
        assert "featured" in data
        assert isinstance(data["featured"], list)

    def test_api_models_model_structure(self, client):
        """Models have required fields."""
        response = client.get("/api/models")

        data = response.json()
        if data["all"]:
            for model in data["all"]:
                assert "id" in model
                assert "provider" in model


class TestAuthKeyEndpoint:
    """Tests for /api/auth/key bootstrap endpoint using TestClient."""

    @pytest.fixture
    def client(self):
        """TestClient with mock agent."""
        from server import app
        from unittest.mock import MagicMock
        import os

        original_bootstrap = os.environ.get("KESTREL_ENABLE_API_KEY_BOOTSTRAP")
        os.environ["KESTREL_ENABLE_API_KEY_BOOTSTRAP"] = "true"
        mock_agent = MagicMock()
        mock_agent.agent_id = "did:test:auth_key"
        mock_agent.storage = None

        original_lifespan = app.router.lifespan_context
        app.router.lifespan_context = _noop_lifespan
        app.state.agent = mock_agent
        try:
            with TestClient(app) as client:
                yield client
        finally:
            if original_bootstrap is None:
                os.environ.pop("KESTREL_ENABLE_API_KEY_BOOTSTRAP", None)
            else:
                os.environ["KESTREL_ENABLE_API_KEY_BOOTSTRAP"] = original_bootstrap
            app.router.lifespan_context = original_lifespan
            app.state.agent = None

    def test_auth_key_rejects_non_localhost(self, client):
        """GET /api/auth/key rejects non-localhost requests (TestClient appears as 'testclient')."""
        response = client.get("/api/auth/key")

        # TestClient appears as "testclient" not "127.0.0.1", so should be rejected
        # This is correct security behavior - auth key bootstrap only from localhost
        assert response.status_code == 403
        assert "localhost" in response.json()["detail"].lower()

    def test_auth_key_disabled_when_oauth_required(self, monkeypatch):
        """GET /api/auth/key returns 404 when OAuth mode disables bootstrap."""
        from server import app
        from unittest.mock import MagicMock

        monkeypatch.setenv("KESTREL_REQUIRE_OAUTH", "true")

        original_lifespan = app.router.lifespan_context
        app.router.lifespan_context = _noop_lifespan
        app.state.agent = MagicMock()

        try:
            with TestClient(app) as client:
                response = client.get("/api/auth/key")
        finally:
            monkeypatch.delenv("KESTREL_REQUIRE_OAUTH", raising=False)
            app.router.lifespan_context = original_lifespan
            app.state.agent = None

        assert response.status_code == 404


class TestProtectedEndpointsRequireAuth:
    """Tests that protected endpoints require authentication using TestClient."""

    @pytest.fixture
    def client(self):
        """TestClient with mock agent."""
        from server import app
        from unittest.mock import MagicMock, AsyncMock

        mock_agent = MagicMock()
        mock_agent.agent_id = "did:test:auth_test"
        mock_agent.storage = MagicMock()
        mock_agent.storage.get_conversations = AsyncMock(return_value=[])
        mock_agent.storage.sovereign_adapter = None

        original_lifespan = app.router.lifespan_context
        app.router.lifespan_context = _noop_lifespan
        app.state.agent = mock_agent
        try:
            with TestClient(app) as client:
                yield client
        finally:
            app.router.lifespan_context = original_lifespan
            app.state.agent = None

    def test_memories_requires_auth(self, client):
        """GET /api/memories requires API key."""
        response = client.get("/api/memories")

        # Should be 401 without auth
        assert response.status_code == 401

    def test_sovereignty_requires_auth(self, client):
        """GET /api/sovereignty/exports requires API key."""
        response = client.get("/api/sovereignty/exports")

        # Should be 401 without auth
        assert response.status_code == 401

    def test_agent_invoke_requires_auth(self, client):
        """POST /agent/invoke requires API key."""
        response = client.post("/agent/invoke", json={"input": "test"})

        # Should be 401 without auth
        assert response.status_code == 401

    def test_health_is_public(self, client):
        """GET /health is public."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_models_requires_auth(self, client):
        """GET /api/models requires API key."""
        response = client.get("/api/models")

        # Should be 401 without auth
        assert response.status_code == 401

    def test_identity_requires_auth(self, client):
        """GET /api/identity requires API key."""
        response = client.get("/api/identity")

        # Should be 401 without auth
        assert response.status_code == 401

    def test_commands_requires_auth(self, client):
        """GET /api/commands requires API key."""
        response = client.get("/api/commands")

        # Should be 401 without auth
        assert response.status_code == 401

    def test_model_current_requires_auth(self, client):
        """GET /api/model/current requires API key."""
        response = client.get("/api/model/current")

        # Should be 401 without auth
        assert response.status_code == 401

    def test_files_requires_auth(self, client):
        """GET /api/files/{hash} requires API key."""
        response = client.get("/api/files/some_hash")

        # Should be 401 without auth
        assert response.status_code == 401

    def test_model_set_requires_auth(self, client):
        """POST /api/model/set requires API key."""
        response = client.post("/api/model/set", json={"model": "gpt-5"})

        # Should be 401 without auth
        assert response.status_code == 401


class TestModelSetEndpoint:
    """Tests for POST /api/model/set endpoint."""

    @pytest.fixture
    def client(self, monkeypatch):
        """TestClient with mock agent for model set testing."""
        from server import app
        from unittest.mock import MagicMock
        from kestrel_sovereign.llm.service import LLMService

        test_api_key = "test-api-key-model-set"
        monkeypatch.setenv("KESTREL_API_KEY", test_api_key)

        mock_agent = MagicMock()
        mock_agent.agent_id = "did:test:model_set"
        mock_agent.llm_service = LLMService()
        mock_agent.storage = None

        original_lifespan = app.router.lifespan_context
        app.router.lifespan_context = _noop_lifespan
        app.state.agent = mock_agent
        try:
            with TestClient(app) as client:
                client.headers.update({"X-API-Key": test_api_key})
                yield client
        finally:
            app.router.lifespan_context = original_lifespan
            app.state.agent = None

    def test_set_model_with_provider(self, client):
        """POST /api/model/set with explicit model and provider."""
        response = client.post("/api/model/set", json={
            "model": "gpt-5-mini",
            "provider": "openai",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model"] == "gpt-5-mini"
        assert data["provider"] == "openai"
        assert data["full_model"] == "openai/gpt-5-mini"

    def test_set_model_combined_format(self, client):
        """POST /api/model/set with combined provider/model format."""
        response = client.post("/api/model/set", json={
            "model": "anthropic/claude-sonnet-4-5",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model"] == "claude-sonnet-4-5"
        assert data["provider"] == "anthropic"

    def test_set_model_without_provider(self, client):
        """POST /api/model/set with model only (auto-detect)."""
        response = client.post("/api/model/set", json={
            "model": "gpt-5-mini",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model"] == "gpt-5-mini"
        assert data["provider"] is None

    def test_set_model_missing_field(self, client):
        """POST /api/model/set without model field returns 400."""
        response = client.post("/api/model/set", json={})

        assert response.status_code == 400

    def test_set_model_persists_to_current(self, client):
        """POST /api/model/set should be reflected in GET /api/model/current."""
        # Set the model
        client.post("/api/model/set", json={
            "model": "gpt-5-mini",
            "provider": "openai",
        })

        # Verify it's reflected
        response = client.get("/api/model/current")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "gpt-5-mini"
        assert data["provider"] == "openai"


class TestChatCompletionsModelPassthrough:
    """Tests that /v1/chat/completions respects the model field."""

    @pytest.fixture
    def client(self, monkeypatch):
        """TestClient with mock agent for chat completions testing."""
        from server import app
        from unittest.mock import MagicMock, AsyncMock

        test_api_key = "test-api-key-chat"
        monkeypatch.setenv("KESTREL_API_KEY", test_api_key)

        mock_agent = MagicMock()
        mock_agent.agent_id = "did:test:chat_completions"
        mock_agent.process_input = AsyncMock(return_value="Test response")
        mock_agent.storage = None

        # Create a mock llm_service with set_model_preference
        mock_llm_service = MagicMock()
        mock_llm_service.set_model_preference = MagicMock()
        mock_llm_service.get_active_model_id = MagicMock(return_value="openai/gpt-5-mini")
        mock_agent.llm_service = mock_llm_service

        original_lifespan = app.router.lifespan_context
        app.router.lifespan_context = _noop_lifespan
        app.state.agent = mock_agent
        try:
            with TestClient(app) as client:
                client.headers.update({"X-API-Key": test_api_key})
                yield client, mock_agent
        finally:
            app.router.lifespan_context = original_lifespan
            app.state.agent = None

    def test_model_passed_to_process_input(self, client):
        """Model from request is passed as model_override to process_input."""
        test_client, mock_agent = client

        response = test_client.post("/v1/chat/completions", json={
            "model": "openai/gpt-5-mini",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert response.status_code == 200
        # Verify process_input was called with model_override
        mock_agent.process_input.assert_called_once()
        call_kwargs = mock_agent.process_input.call_args
        assert call_kwargs.kwargs.get("model_override") == "openai/gpt-5-mini"

    def test_model_is_not_persisted_via_set_preference(self, client):
        """Model from request should stay request-scoped for OpenAI-compatible calls."""
        test_client, mock_agent = client

        test_client.post("/v1/chat/completions", json={
            "model": "openai/gpt-5-mini",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        mock_agent.llm_service.set_model_preference.assert_not_called()

    def test_kestrel_local_model_not_overridden(self, client):
        """The default 'kestrel-local' model should not override."""
        test_client, mock_agent = client

        test_client.post("/v1/chat/completions", json={
            "model": "kestrel-local",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        mock_agent.llm_service.set_model_preference.assert_not_called()
        call_kwargs = mock_agent.process_input.call_args
        assert call_kwargs.kwargs.get("model_override") is None

    def test_no_model_field_uses_default(self, client):
        """When no model field is sent, don't override anything."""
        test_client, mock_agent = client

        test_client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
        })

        mock_agent.llm_service.set_model_preference.assert_not_called()
        call_kwargs = mock_agent.process_input.call_args
        assert call_kwargs.kwargs.get("model_override") is None

    def test_response_echoes_model(self, client):
        """Response should echo the model from the request."""
        test_client, mock_agent = client

        response = test_client.post("/v1/chat/completions", json={
            "model": "openai/gpt-5-mini",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "openai/gpt-5-mini"
