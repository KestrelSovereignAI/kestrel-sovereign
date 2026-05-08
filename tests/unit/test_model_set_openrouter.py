"""
Unit tests for model-set command with OpenRouter models.

OpenRouter models have format "provider/model" (e.g., "google/gemini-3-pro-preview")
but the actual provider should be "openrouter", not "google".
"""
import json
from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.model.feature import ModelAgent
from kestrel_sovereign.llm.model_cache import get_shared_model_cache
from kestrel_sovereign.llm.model_metadata import ModelInfo


class TestModelSetOpenRouter:
    """Test that model-set correctly identifies OpenRouter models."""

    @pytest.fixture(autouse=True)
    def clear_shared_model_cache(self):
        cache = get_shared_model_cache()
        cache.clear()
        yield
        cache.clear()

    async def _feature(self) -> tuple[ModelAgent, MagicMock]:
        llm_service = MagicMock()
        llm_service.get_model_preference.return_value = {"model": "auto"}
        agent = SimpleNamespace(llm_service=llm_service, features={})
        feature = ModelAgent(agent)
        await feature.initialize()
        return feature, llm_service

    def _model_changed_payload(self, message: str) -> dict:
        marker = "MODEL_CHANGED:"
        assert marker in message
        return json.loads(message.split(marker, 1)[1])

    @pytest.mark.asyncio
    async def test_cached_openrouter_model_keeps_full_id_and_uses_openrouter(self):
        """OpenRouter-hosted vendor/model IDs must not be split as direct vendors."""
        get_shared_model_cache().set([
            ModelInfo(
                id="google/gemini-3-pro-preview",
                provider="openrouter",
                display_name="Gemini 3 Pro Preview",
            ),
        ])
        feature, llm_service = await self._feature()

        result = await feature.set_model("google/gemini-3-pro-preview")

        assert result.status is ToolResultStatus.OK
        assert result.data["vendor"] == "openrouter"
        assert result.data["model_name"] == "google/gemini-3-pro-preview"
        assert result.data["model"] == "openrouter/google/gemini-3-pro-preview"
        llm_service.set_model_preference.assert_called_once_with(
            "google/gemini-3-pro-preview",
            "openrouter",
            None,
        )
        assert self._model_changed_payload(result.data["message"]) == {
            "model": "openrouter/google/gemini-3-pro-preview",
            "vendor": "openrouter",
            "route": None,
            "model_name": "google/gemini-3-pro-preview",
        }

    @pytest.mark.asyncio
    async def test_openrouter_only_vendor_fallback_uses_openrouter_without_cache(self):
        """Known OpenRouter-only prefixes still route through OpenRouter before discovery."""
        feature, llm_service = await self._feature()

        result = await feature.set_model("meta-llama/llama-3.3-70b-instruct")

        assert result.status is ToolResultStatus.OK
        assert result.data["vendor"] == "openrouter"
        assert result.data["model_name"] == "meta-llama/llama-3.3-70b-instruct"
        llm_service.set_model_preference.assert_called_once_with(
            "meta-llama/llama-3.3-70b-instruct",
            "openrouter",
            None,
        )

    @pytest.mark.asyncio
    async def test_direct_provider_model_is_split_into_provider_and_model(self):
        """Direct provider models keep the provider prefix as routing metadata."""
        feature, llm_service = await self._feature()

        result = await feature.set_model("openai/gpt-5")

        assert result.status is ToolResultStatus.OK
        assert result.data["vendor"] == "openai"
        assert result.data["model_name"] == "gpt-5"
        assert result.data["model"] == "openai/gpt-5"
        llm_service.set_model_preference.assert_called_once_with("gpt-5", "openai", None)

    @pytest.mark.asyncio
    async def test_bare_model_has_no_vendor(self):
        """Bare model IDs should not invent routing metadata."""
        feature, llm_service = await self._feature()

        result = await feature.set_model("gpt-5")

        assert result.status is ToolResultStatus.OK
        assert result.data["vendor"] is None
        assert result.data["model_name"] == "gpt-5"
        assert result.data["model"] == "gpt-5"
        llm_service.set_model_preference.assert_called_once_with("gpt-5", None, None)
