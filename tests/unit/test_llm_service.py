"""
Unit tests for LLMService core methods.

Tests the unified LLM provider manager with mocked dependencies.
No real API calls are made.
"""

import asyncio
import os
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from pydantic import BaseModel

from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from kestrel_sovereign.llm.service import (
    BackendType,
    LLMService,
    LLMServiceError,
    RemoteGPUConfig,
)
from kestrel_sovereign.llm.error_handling import LLMAllProvidersFailedError, LLMProviderError


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Mock configuration for LLMService."""
    return {
        "provider_priority": ["openai", "anthropic"],
        "openai": {
            "api_key": "sk-test-key",
            "model": "gpt-5-mini",
            "base_url": "https://api.openai.com/v1",
        },
        "anthropic": {
            "api_key": "sk-ant-test-key",
            "model": "claude-sonnet-4-5",
            "base_url": "https://api.anthropic.com/v1",
        },
    }


@pytest.fixture
def mock_mandate_config():
    """Mock model mandate configuration."""
    return {
        "defaults": {
            "preferred": "",
            "cheap_model": "auto",
            "cheap_model_hints": ["haiku", "mini", "flash"],
            "banned": ["gpt-3"],
        },
        "mandates": {
            "vision": "openai",
            "code": "anthropic",
        },
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI async client."""
    client = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_adapter():
    """Mock LLM adapter."""
    adapter = Mock()
    adapter.create_messages = Mock(return_value=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    ])
    adapter.get_response = AsyncMock(return_value=LLMResponse(
        content="Hello! How can I help you?",
        input_tokens=10,
        output_tokens=8,
        total_tokens=18,
    ))
    return adapter


@pytest.fixture
def mock_provider_registry(mock_openai_client, mock_adapter):
    """Mock ProviderRegistry."""
    registry = Mock()

    # Mock ProviderInfo objects. Under the vendor/route architecture each
    # entry represents a route; `name` is the composite "vendor:route" key,
    # `vendor` carries the grouping dimension, `route` the per-route identity.
    provider_info_openai = Mock(spec=[
        "name", "vendor", "route", "client", "adapter", "model",
        "is_cloud", "is_local", "base_url", "selection_hints",
    ])
    provider_info_openai.name = "openai:api"
    provider_info_openai.vendor = "openai"
    provider_info_openai.route = "api"
    provider_info_openai.client = mock_openai_client
    provider_info_openai.adapter = mock_adapter
    provider_info_openai.model = "gpt-5-mini"
    provider_info_openai.is_cloud = True
    provider_info_openai.is_local = False
    provider_info_openai.base_url = None
    provider_info_openai.selection_hints = []

    provider_info_anthropic = Mock(spec=[
        "name", "vendor", "route", "client", "adapter", "model",
        "is_cloud", "is_local", "base_url", "selection_hints",
    ])
    provider_info_anthropic.name = "anthropic:api"
    provider_info_anthropic.vendor = "anthropic"
    provider_info_anthropic.route = "api"
    provider_info_anthropic.client = AsyncMock()
    provider_info_anthropic.adapter = mock_adapter
    provider_info_anthropic.model = "claude-sonnet-4-5"
    provider_info_anthropic.is_cloud = True
    provider_info_anthropic.is_local = False
    provider_info_anthropic.base_url = None
    provider_info_anthropic.selection_hints = []

    provider_info_cheap = Mock(spec=[
        "name", "vendor", "route", "client", "adapter", "model",
        "is_cloud", "is_local", "base_url", "selection_hints",
    ])
    provider_info_cheap.name = "anthropic:api"
    provider_info_cheap.vendor = "anthropic"
    provider_info_cheap.route = "api"
    provider_info_cheap.client = AsyncMock()
    provider_info_cheap.adapter = mock_adapter
    provider_info_cheap.model = "claude-haiku-4-5"
    provider_info_cheap.is_cloud = True
    provider_info_cheap.is_local = False
    provider_info_cheap.base_url = None
    provider_info_cheap.selection_hints = []

    registry.initialize_providers = Mock(return_value=[
        provider_info_openai,
        provider_info_anthropic,
    ])
    registry.get_provider_by_name = Mock(return_value=provider_info_anthropic)
    registry.get_providers_with_pattern = Mock(return_value=[provider_info_cheap])
    registry.update_provider_client = Mock(return_value=True)

    return registry


@pytest_asyncio.fixture
async def llm_service(mock_config, mock_mandate_config, mock_provider_registry):
    """Create LLMService with mocked dependencies."""
    with patch("kestrel_sovereign.llm.service.load_config") as mock_load_config, \
         patch("kestrel_sovereign.llm.service.ProviderRegistry") as mock_registry_class:

        # Setup config mocking
        mock_load_config.side_effect = lambda path: (
            mock_config if "llm_config" in path else mock_mandate_config
        )

        # Setup registry mocking
        mock_registry_class.return_value = mock_provider_registry

        service = LLMService(config_path="llm_config.toml")

        # Mock the usage tracking and constitutional profiles (which set attributes)
        service._usage_db = None
        service._db_initialized = False
        service._usage_database_url = None
        service._db_backend = "sqlite"

        yield service

        # Cleanup
        await service.close()


# =============================================================================
# Priority 1 — Model Preference Tests
# =============================================================================


class TestModelPreference:
    """Tests for model preference methods."""

    @pytest.mark.asyncio
    async def test_set_model_preference_with_vendor(self, llm_service):
        """Setting with an explicit vendor stores it in the mandate."""
        llm_service.set_model_preference("gpt-5", vendor="openai")

        pref = llm_service.get_model_preference()
        assert pref["model"] == "gpt-5"
        assert pref["vendor"] == "openai"
        assert pref["route"] is None

    @pytest.mark.asyncio
    async def test_set_model_preference_with_vendor_and_route(self, llm_service):
        """Route narrows routing to the exact <vendor>:<route> entry."""
        llm_service.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        pref = llm_service.get_model_preference()
        assert pref["vendor"] == "anthropic"
        assert pref["route"] == "plan"
        assert pref["model"] == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_set_model_preference_without_vendor(self, llm_service):
        """Setting model without vendor leaves vendor auto-detect (None)."""
        llm_service.set_model_preference("claude-sonnet-4-5")

        pref = llm_service.get_model_preference()
        assert pref["model"] == "claude-sonnet-4-5"
        assert pref["vendor"] is None
        assert pref["route"] is None

    @pytest.mark.asyncio
    async def test_clear_model_preference(self, llm_service):
        """Clearing returns all three slots to None."""
        llm_service.set_model_preference("gpt-5", vendor="openai")
        llm_service.clear_model_preference()

        pref = llm_service.get_model_preference()
        assert pref["model"] is None
        assert pref["vendor"] is None
        assert pref["route"] is None

    @pytest.mark.asyncio
    async def test_model_preference_persistence_tasks_are_owned(self, llm_service):
        """Preference persistence is scheduled through owned service tasks."""
        calls = []

        async def persist(model, vendor, route):
            calls.append((model, vendor, route))

        llm_service.set_preference_persistence_callback(persist)
        llm_service.set_model_preference("gpt-5", vendor="openai")

        assert len(llm_service._preference_persistence_tasks) == 1

        await llm_service.drain_preference_persistence()

        assert calls == [("gpt-5", "openai", None)]
        assert llm_service._preference_persistence_tasks == set()

    @pytest.mark.asyncio
    async def test_close_waits_for_preference_persistence(self, llm_service):
        """close() waits for pending preference persistence before cleanup returns."""
        calls = []
        started = asyncio.Event()
        release = asyncio.Event()

        async def persist(model, vendor, route):
            calls.append(("start", model, vendor, route))
            started.set()
            await release.wait()
            calls.append(("done", model, vendor, route))

        llm_service.providers = []
        llm_service.set_preference_persistence_callback(persist)
        llm_service.set_model_preference("gpt-5", vendor="openai")
        await started.wait()

        close_task = asyncio.create_task(llm_service.close())
        await asyncio.sleep(0)

        assert not close_task.done()

        release.set()
        await close_task

        assert calls == [
            ("start", "gpt-5", "openai", None),
            ("done", "gpt-5", "openai", None),
        ]
        assert llm_service._preference_persistence_tasks == set()

    @pytest.mark.asyncio
    async def test_get_model_preference_default(self, llm_service):
        """Default preference has None for all three slots (vendor, model, route)."""
        pref = llm_service.get_model_preference()
        assert isinstance(pref, dict)
        assert set(pref.keys()) == {"vendor", "model", "route"}
        assert pref["vendor"] is None
        assert pref["model"] is None
        assert pref["route"] is None

    @pytest.mark.asyncio
    async def test_get_cheap_model_from_config(self, llm_service):
        """Test get_cheap_model resolves configured selector via discovery-backed hints."""
        cheap_model = llm_service.get_cheap_model()
        assert cheap_model == "claude-haiku-4-5"

    @pytest.mark.asyncio
    async def test_get_cheap_model_returns_none_without_config_hints(self, llm_service, mock_provider_registry):
        """Test get_cheap_model requires config hints instead of hidden code defaults."""
        # Clear the config
        llm_service.mandate_config = {"defaults": {}}

        cheap_model = llm_service.get_cheap_model()
        assert cheap_model is None


# =============================================================================
# Priority 2 — Core Generation Tests
# =============================================================================


class TestCoreGeneration:
    """Tests for core generation methods."""

    @pytest.mark.asyncio
    async def test_get_response_basic(self, llm_service):
        """Test basic get_response with mocked provider."""
        response = await llm_service.get_response(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello, world!",
        )

        assert isinstance(response, str)
        assert "Hello" in response or "help" in response

    @pytest.mark.asyncio
    async def test_get_response_with_tools(self, llm_service, mock_adapter):
        """Test get_response with tool calling."""
        # Mock response with tool calls
        mock_adapter.get_response = AsyncMock(return_value=LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments={"city": "NYC"})
            ],
            input_tokens=20,
            output_tokens=15,
        ))

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        response = await llm_service.get_response(
            system_prompt="You have access to tools.",
            user_prompt="What's the weather in NYC?",
            tools=tools,
        )

        assert isinstance(response, LLMResponse)
        assert response.has_tool_calls

    @pytest.mark.asyncio
    async def test_get_response_provider_fallback(self, llm_service, mock_adapter):
        """Test provider fallback when first provider fails."""
        # First call fails, second succeeds
        mock_adapter.get_response = AsyncMock(
            side_effect=[
                LLMProviderError("openai", "OpenAI failed"),
                LLMResponse(content="Success from fallback", input_tokens=10, output_tokens=8),
            ]
        )

        response = await llm_service.get_response(
            system_prompt="Test",
            user_prompt="Test prompt",
        )

        assert isinstance(response, str)
        assert "Success" in response

    @pytest.mark.asyncio
    async def test_get_response_all_providers_fail(self, llm_service, mock_adapter):
        """Test that exception is raised when all providers fail."""
        mock_adapter.get_response = AsyncMock(
            side_effect=LLMProviderError("openai", "Provider failed")
        )

        with pytest.raises(LLMAllProvidersFailedError):
            await llm_service.get_response(
                system_prompt="Test",
                user_prompt="Test prompt",
            )

    @pytest.mark.asyncio
    async def test_get_response_with_model_override(self, llm_service):
        """Test get_response with explicit model override."""
        response = await llm_service.get_response(
            system_prompt="Test",
            user_prompt="Test prompt",
            model_override="gpt-5",
        )

        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_get_response_force_local_only(self, llm_service):
        """Test get_response with force_local_only flag."""
        # Add a local route — the force_local_only filter now uses is_local.
        local_provider = {
            "name": "ollama:local",
            "vendor": "ollama",
            "route": "local",
            "client": AsyncMock(),
            "adapter": Mock(),
            "model": "llama3.2:3b",
            "is_cloud": False,
            "is_local": True,
            "base_url": None,
            "selection_hints": [],
        }
        local_provider["adapter"].create_messages = Mock(return_value=[])
        local_provider["adapter"].get_response = AsyncMock(
            return_value=LLMResponse(content="Local response", input_tokens=5, output_tokens=5)
        )
        llm_service.providers.append(local_provider)

        response = await llm_service.get_response(
            system_prompt="Test",
            user_prompt="Test prompt",
            force_local_only=True,
        )

        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_get_response_with_structured_output(self, llm_service, mock_adapter):
        """Test get_response with Pydantic response format."""
        class TestOutput(BaseModel):
            result: str
            confidence: float

        mock_adapter.get_response = AsyncMock(return_value=LLMResponse(
            content='{"result": "test", "confidence": 0.9}',
            input_tokens=10,
            output_tokens=8,
        ))

        response = await llm_service.get_response(
            system_prompt="Test",
            user_prompt="Test prompt",
            response_format=TestOutput,
        )

        assert isinstance(response, LLMResponse)

    @pytest.mark.asyncio
    async def test_get_response_with_model(self, llm_service):
        """Test get_response_with_model using specific model."""
        response = await llm_service.get_response_with_model(
            model_id="gpt-5-mini",
            system_prompt="Test",
            user_prompt="Hello",
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_get_audit_response(self, llm_service, mock_adapter):
        """Test get_audit_response returns structured audit result without a dedicated audit model."""
        # Mock JSON response
        mock_adapter.get_response = AsyncMock(return_value=LLMResponse(
            content='{"risk_level": 1, "reasoning": "Normal response"}',
            input_tokens=50,
            output_tokens=20,
        ))

        audit_result = await llm_service.get_audit_response(
            text_to_audit="This is a normal helpful response."
        )

        assert isinstance(audit_result, dict)
        assert "risk_level" in audit_result
        assert "reasoning" in audit_result

    @pytest.mark.asyncio
    async def test_generate_basic(self, llm_service):
        """Test generate method (lower-level generation)."""
        response = await llm_service.generate(
            system_prompt="You are helpful",
            user_prompt="Say hello",
        )

        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_generate_with_messages(self, llm_service):
        """Test generate_with_messages for multi-turn conversations."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        response = await llm_service.generate_with_messages(messages=messages)

        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_generate_with_model_mandate(self, llm_service):
        """Test that model mandate preference is respected in generate_with_messages."""
        llm_service.set_model_preference("claude-sonnet-4-5", vendor="anthropic")

        messages = [{"role": "user", "content": "Test"}]
        response = await llm_service.generate_with_messages(messages=messages)

        assert response is not None


# =============================================================================
# Priority 3 — Backend and Lifecycle Tests
# =============================================================================


class TestBackendLifecycle:
    """Tests for backend switching and lifecycle methods."""

    @pytest.mark.asyncio
    async def test_switch_backend_to_cloud(self, llm_service):
        """Test switching to cloud backend."""
        llm_service.switch_backend(BackendType.CLOUD)

        status = llm_service.get_backend_status()
        assert status["current_backend"] == "cloud"

    @pytest.mark.asyncio
    async def test_switch_backend_to_local(self, llm_service):
        """Test switching to local backend."""
        llm_service.switch_backend(BackendType.LOCAL)

        status = llm_service.get_backend_status()
        assert status["current_backend"] == "local"

    @pytest.mark.asyncio
    async def test_switch_backend_to_remote_gpu(self, llm_service):
        """Test switching to remote GPU backend."""
        config = {
            "base_url": "https://gpu.example.com/v1",
            "model": "llama-70b",
            "api_key": "test-key",
        }

        llm_service.switch_backend(BackendType.REMOTE_GPU, config=config)

        status = llm_service.get_backend_status()
        assert status["current_backend"] == "remote_gpu"
        assert status["remote_active"] is True

    @pytest.mark.asyncio
    async def test_switch_backend_remote_gpu_requires_config(self, llm_service):
        """Test that remote GPU backend requires configuration."""
        with pytest.raises(LLMServiceError, match="requires configuration"):
            llm_service.switch_backend(BackendType.REMOTE_GPU)

    @pytest.mark.asyncio
    async def test_get_backend_status_structure(self, llm_service):
        """Test backend status returns correct structure."""
        status = llm_service.get_backend_status()

        assert "current_backend" in status
        assert "default_backend" in status
        assert "remote_active" in status
        assert "remote_metadata" in status
        assert "last_remote_error" in status

    @pytest.mark.asyncio
    async def test_close_providers(self, llm_service, mock_openai_client):
        """Test that close() properly closes all provider clients."""
        await llm_service.close()

        # Verify close was called on clients
        assert mock_openai_client.close.called

    @pytest.mark.asyncio
    async def test_close_handles_exceptions_gracefully(self, llm_service):
        """Test that close() handles exceptions during cleanup."""
        # Add a provider with a broken close method
        broken_client = AsyncMock()
        broken_client.close = AsyncMock(side_effect=Exception("Close failed"))

        llm_service.providers.append({
            "name": "broken",
            "client": broken_client,
            "adapter": Mock(),
            "model": "test",
        })

        # Should not raise
        await llm_service.close()

    @pytest.mark.asyncio
    async def test_close_accepts_sync_provider_close(self, llm_service, caplog):
        """Provider clients may expose synchronous close() methods."""
        sync_client = Mock()
        sync_client.close = Mock(return_value=None)

        llm_service.providers = [{
            "name": "sync_provider",
            "client": sync_client,
            "adapter": Mock(),
            "model": "test",
        }]

        caplog.set_level("WARNING")
        await llm_service.close()

        sync_client.close.assert_called_once()
        assert "Unexpected error closing sync_provider client" not in caplog.text

    @pytest.mark.asyncio
    async def test_set_observability_store(self, llm_service):
        """Test setting observability store."""
        mock_store = Mock()
        llm_service.set_observability_store(mock_store)

        assert llm_service._observability_store is mock_store

    @pytest.mark.asyncio
    async def test_set_observability_context(self, llm_service):
        """Test setting observability context."""
        llm_service.set_observability_context(
            session_id="sess-123",
            companion_id="comp-456",
            user_id="user-789",
        )

        assert llm_service._observability_context["session_id"] == "sess-123"
        assert llm_service._observability_context["companion_id"] == "comp-456"
        assert llm_service._observability_context["user_id"] == "user-789"

    @pytest.mark.asyncio
    async def test_observability_logging_on_success(self, llm_service):
        """Test that successful LLM calls are logged to observability store."""
        mock_store = AsyncMock()
        llm_service.set_observability_store(mock_store)
        llm_service.set_observability_context(session_id="test-session")

        await llm_service.get_response(
            system_prompt="Test",
            user_prompt="Hello",
        )

        # Verify observability store was called
        assert mock_store.log_llm_call.called


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_no_providers_initialized(self, mock_config, mock_mandate_config):
        """Test that RuntimeError is raised when no providers are initialized."""
        with patch("kestrel_sovereign.llm.service.load_config") as mock_load_config, \
             patch("kestrel_sovereign.llm.service.ProviderRegistry") as mock_registry_class:

            mock_load_config.side_effect = lambda path: (
                mock_config if "llm_config" in path else mock_mandate_config
            )

            # Mock empty providers
            mock_registry = Mock()
            mock_registry.initialize_providers = Mock(return_value=[])
            mock_registry_class.return_value = mock_registry

            service = LLMService(config_path="llm_config.toml")

            # Mock the usage tracking attributes
            service._usage_db = None
            service._db_initialized = False
            service._usage_database_url = None
            service._db_backend = "sqlite"

            with pytest.raises(RuntimeError, match="No LLM providers"):
                await service.get_response(
                    system_prompt="Test",
                    user_prompt="Test",
                )

            await service.close()

    @pytest.mark.asyncio
    async def test_model_mandate_keyword_trigger(self, llm_service):
        """Test that model mandate is triggered by keywords in prompt."""
        response = await llm_service.get_response(
            system_prompt="Test",
            user_prompt="Write some code for me",  # "code" should trigger mandate
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_remote_gpu_with_ttl_expiry(self, llm_service):
        """Test that expired remote GPU sessions are deactivated."""
        from datetime import datetime, timedelta, timezone

        config = {
            "base_url": "https://gpu.example.com/v1",
            "model": "llama-70b",
            "expires_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        }

        llm_service.switch_backend(BackendType.REMOTE_GPU, config=config)

        # Try to use expired backend
        with pytest.raises(LLMServiceError, match="expired"):
            llm_service._ensure_remote_active()

    @pytest.mark.asyncio
    async def test_generate_with_remote_gpu_fallback(self, llm_service, mock_adapter):
        """Test that generate falls back to providers when remote GPU fails."""
        # Setup remote GPU backend
        config = {
            "base_url": "https://gpu.example.com/v1",
            "model": "llama-70b",
        }
        llm_service.switch_backend(BackendType.REMOTE_GPU, config=config)

        # Mock remote adapter to fail
        llm_service._remote_adapter.get_response = AsyncMock(
            side_effect=Exception("Remote GPU unavailable")
        )

        # Should fall back to regular providers
        response = await llm_service.generate(
            system_prompt="Test",
            user_prompt="Hello",
        )

        assert isinstance(response, str)
        # Backend should be deactivated after failure
        status = llm_service.get_backend_status()
        assert not status["remote_active"]


# =============================================================================
# Priority 5 — Streaming Mandate Preference Tests
# =============================================================================


class TestStreamingMandatePreference:
    """Tests that streaming methods respect mandate preference."""

    @pytest.mark.asyncio
    async def test_get_streaming_response_uses_mandate(self, llm_service, mock_adapter):
        """Test that get_streaming_response respects mandate preference."""
        # Set mandate to anthropic/claude-sonnet-4-5
        llm_service.set_model_preference("claude-sonnet-4-5", vendor="anthropic")

        # Mock streaming response
        async def mock_streaming(*args, **kwargs):
            yield "Hello "
            yield "world"
        mock_adapter.get_streaming_response = Mock(return_value=mock_streaming())

        chunks = []
        async for chunk in llm_service.get_streaming_response(
            system_prompt="Test",
            user_prompt="Hello",
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Verify the adapter was called with the mandated model
        call_args = mock_adapter.get_streaming_response.call_args
        assert call_args is not None
        assert call_args.kwargs.get("model") == "claude-sonnet-4-5" or \
               (len(call_args.args) > 1 and call_args.args[1] == "claude-sonnet-4-5")

    @pytest.mark.asyncio
    async def test_stream_with_messages_uses_mandate(self, llm_service, mock_adapter):
        """Test that stream_with_messages respects mandate preference."""
        # Set mandate to anthropic/claude-sonnet-4-5
        llm_service.set_model_preference("claude-sonnet-4-5", vendor="anthropic")

        # Mock streaming response
        async def mock_streaming(*args, **kwargs):
            yield "Streamed "
            yield "response"
        mock_adapter.get_streaming_response = Mock(return_value=mock_streaming())

        messages = [{"role": "user", "content": "Test"}]
        chunks = []
        async for chunk in llm_service.stream_with_messages(
            messages=messages,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Verify the adapter was called with the mandated model
        call_args = mock_adapter.get_streaming_response.call_args
        assert call_args is not None
        assert call_args.kwargs.get("model") == "claude-sonnet-4-5" or \
               (len(call_args.args) > 1 and call_args.args[1] == "claude-sonnet-4-5")

    @pytest.mark.asyncio
    async def test_streaming_without_mandate_uses_default(self, llm_service, mock_adapter):
        """Test that streaming without mandate uses provider default model."""
        # Ensure no mandate is set
        llm_service.clear_model_preference()

        async def mock_streaming(*args, **kwargs):
            yield "Default response"
        mock_adapter.get_streaming_response = Mock(return_value=mock_streaming())

        messages = [{"role": "user", "content": "Test"}]
        chunks = []
        async for chunk in llm_service.stream_with_messages(
            messages=messages,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Should use the provider's default model (gpt-5-mini for first provider)
        call_args = mock_adapter.get_streaming_response.call_args
        assert call_args is not None
        model_used = call_args.kwargs.get("model") or (call_args.args[1] if len(call_args.args) > 1 else None)
        assert model_used == "gpt-5-mini"

    @pytest.mark.asyncio
    async def test_streaming_override_takes_precedence_over_mandate(self, llm_service, mock_adapter):
        """Test that explicit model_override takes precedence over mandate."""
        # Set mandate to anthropic
        llm_service.set_model_preference("claude-sonnet-4-5", vendor="anthropic")

        async def mock_streaming(*args, **kwargs):
            yield "Override response"
        mock_adapter.get_streaming_response = Mock(return_value=mock_streaming())

        messages = [{"role": "user", "content": "Test"}]
        chunks = []
        async for chunk in llm_service.stream_with_messages(
            messages=messages,
            model_override="openai/gpt-5-mini",
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Should use the override model, not the mandate
        # stream_with_messages resolves "openai/gpt-5-mini" → model "gpt-5-mini"
        call_args = mock_adapter.get_streaming_response.call_args
        assert call_args is not None
        model_used = call_args.kwargs.get("model") or (call_args.args[1] if len(call_args.args) > 1 else None)
        assert model_used in ("gpt-5-mini", "openai/gpt-5-mini")
