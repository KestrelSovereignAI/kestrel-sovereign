"""
LLM Service - Unified LLM provider management with remote GPU support.

This is the single entry point for all LLM operations. It handles:
- Multiple provider initialization and fallback
- Remote GPU backend switching (RunPod, etc.)
- Model mandate routing
- Usage tracking
"""
import logging
import re
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Type, AsyncIterator, TYPE_CHECKING

import openai

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase
from dotenv import load_dotenv
from pydantic import BaseModel

from .provider_registry import ProviderRegistry, ProviderInfo, ProviderInitializationError
from .error_handling import (
    handle_llm_errors,
    handle_observability_errors,
    handle_storage_errors,
    LLMError,
    LLMProviderError,
    LLMAllProvidersFailedError
)
from .openai_adapter import OpenAIAdapter
from .adapter import LLMResponse
from .model_discovery import ModelDiscoveryMixin
from .mandate import ModelMandateMixin
from .usage_tracking import UsageTrackingMixin
from kestrel_sovereign.kestrel_config.constants import (
    HTTP_TIMEOUT_MEDIUM,
    CLIENT_CLOSE_TIMEOUT,
)
from kestrel_sovereign.config import load_config

logger = logging.getLogger(__name__)


class BackendType(str, Enum):
    """LLM backend types."""
    CLOUD = "cloud"
    LOCAL = "local"
    REMOTE_GPU = "remote_gpu"


@dataclass
class RemoteGPUConfig:
    """Configuration for remote GPU backend (RunPod, etc.)."""
    base_url: str
    model: str
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    context_window: Optional[int] = None
    ttl_seconds: Optional[int] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = HTTP_TIMEOUT_MEDIUM


class LLMServiceError(LLMError):
    """Raised when LLM service cannot fulfill a request."""


class LLMService(ModelDiscoveryMixin, ModelMandateMixin, UsageTrackingMixin):
    """Unified LLM service with provider fallback and remote GPU support."""

    def __init__(self, config_path: str = "llm_config.toml", database_url: Optional[str] = None):
        """Initialize LLM service.

        Args:
            config_path: Path to LLM configuration file.
            database_url: Optional PostgreSQL connection URL for usage tracking.
                         If provided, uses PostgreSQL. Otherwise checks env vars,
                         then falls back to SQLite.
        """
        load_dotenv()

        # default_model is derived from first provider in provider_priority
        # (see provider_registry and endpoints/models.py)
        self.default_model = None  # Deprecated: use providers[0] instead
        self.config = load_config(config_path)
        self.mandate_config = load_config("model_mandate.toml")

        # Initialize provider registry
        self.provider_registry = ProviderRegistry(self.config)
        try:
            self.providers = self._convert_providers_format(self.provider_registry.initialize_providers())
        except ProviderInitializationError as e:
            logger.error(f"Failed to initialize providers: {e}")
            self.providers = []

        # Model discovery cache
        self._model_cache = None
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5 minutes

        # Storage info cache
        self._storage_cache = None
        self._storage_cache_timestamp = None
        self._storage_cache_ttl = 60  # 1 minute

        # Database for model usage tracking (uses abstract data layer)
        self._init_usage_tracking(database_url)

        # Runtime mandate state
        self._mandate_preference = {"model": None, "provider": None}
        self._mandate_fallbacks = []

        # Remote GPU backend state (merged from BrainRouter)
        self._backend = BackendType.CLOUD
        self._default_backend = BackendType.CLOUD
        self._remote_config: Optional[RemoteGPUConfig] = None
        self._remote_client: Optional[openai.AsyncOpenAI] = None
        self._remote_adapter = OpenAIAdapter()
        self._last_remote_error: Optional[str] = None

        # Observability store for logging LLM calls (A2A-compatible)
        # Set via set_observability_store() after initialization
        self._observability_store = None
        self._observability_context: Dict[str, Any] = {}

        # Metering callback for usage billing (Vending Machine)
        # Set via set_metering_callback() after initialization
        self._metering_callback = None

    def _convert_providers_format(self, provider_infos: List[ProviderInfo]) -> List[Dict[str, Any]]:
        """Convert ProviderInfo objects to legacy dictionary format.

        This maintains backward compatibility while using the new ProviderRegistry.

        Args:
            provider_infos: List of ProviderInfo objects from registry

        Returns:
            List of provider dictionaries in legacy format
        """
        return [
            {
                "name": provider.name,
                "client": provider.client,
                "adapter": provider.adapter,
                "model": provider.model
            }
            for provider in provider_infos
        ]

    def _initialize_providers(self) -> List[Dict[str, Any]]:
        """Initialize provider clients and adapters based on config file.

        This method is deprecated. Provider initialization is now handled by ProviderRegistry.
        This method is kept for backward compatibility and now delegates to the registry.
        """
        try:
            provider_infos = self.provider_registry.initialize_providers()
            return self._convert_providers_format(provider_infos)
        except ProviderInitializationError as e:
            logger.error(f"Failed to initialize providers: {e}")
            return []

    async def use_agent_key(
        self,
        agent_did: str,
        db: "AsyncDatabase",
        provider: str = "openrouter",
    ) -> bool:
        """
        Switch to using an agent's provisioned API key.

        This replaces the shared key with the agent's own key for billing isolation.
        The agent's key was created at inception and stored encrypted.

        Args:
            agent_did: The agent's DID
            db: Database connection for ServiceKeyStorage
            provider: Provider name (default: openrouter)

        Returns:
            True if key was activated, False if agent has no key

        Raises:
            KeyNotConfiguredError: If key retrieval fails
        """
        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage, KeyNotConfiguredError

        try:
            key_storage = ServiceKeyStorage(db, agent_did)
            agent_key = await key_storage.get_key(provider_id=provider)
        except KeyNotConfiguredError:
            logger.debug(f"Agent {agent_did[:20]}... has no {provider} key, using shared key")
            return False

        # Get base_url from config
        provider_config = self.config.get(provider, {})
        base_url = provider_config.get("base_url")

        if not base_url:
            logger.warning(f"No base_url for provider {provider}")
            return False

        # Create new client with agent's key
        new_client = openai.AsyncOpenAI(api_key=agent_key, base_url=base_url)

        # Update the provider's client through registry
        if self.provider_registry.update_provider_client(provider, new_client):
            # Also update the legacy format for backward compatibility
            for p in self.providers:
                if p["name"] == provider:
                    p["client"] = new_client
                    break
            logger.info(f"Activated agent key for {provider} (DID: {agent_did[:20]}...)")
            return True

        logger.warning(f"Provider {provider} not found in initialized providers")
        return False

    def _get_model_for_prompt(self, user_prompt: str) -> Optional[str]:
        """Determine best model based on user prompt and mandate rules."""
        mandates = self.mandate_config.get("mandates", {})
        prompt_lower = user_prompt.lower()

        for keyword, model in mandates.items():
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, prompt_lower):
                banned_list = self.mandate_config.get("defaults", {}).get("banned", [])
                if model in banned_list or any(b in model for b in banned_list):
                    logger.warning(f"Mandate model '{model}' is banned. Ignoring.")
                    continue
                logger.info(f"Model mandate triggered by '{keyword}'. Using: {model}")
                return model

        return self.mandate_config.get("defaults", {}).get("preferred")

    # ==================== Observability Methods ====================

    def set_observability_store(self, store) -> None:
        """Set the observability store for logging LLM calls.

        Args:
            store: ObservabilityStore instance from kestrel_sovereign.a2a.stores.unified
        """
        self._observability_store = store
        logger.info("LLM observability enabled")

    def set_metering_callback(self, callback) -> None:
        """Set the metering callback for usage billing (Vending Machine).

        The callback will be called with:
            await callback(
                companion_id=str,
                user_id=str,
                provider=str,
                model=str,
                prompt_tokens=int,
                completion_tokens=int,
            )

        Args:
            callback: Async function to call after each LLM call
        """
        self._metering_callback = callback
        logger.info("LLM metering enabled")

    def set_observability_context(
        self,
        session_id: Optional[str] = None,
        companion_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """Set context for observability logging (called per-request).

        Args:
            session_id: A2A session ID
            companion_id: Companion UUID
            user_id: User UUID
        """
        self._observability_context = {
            "session_id": session_id,
            "companion_id": companion_id,
            "user_id": user_id,
        }

    @handle_observability_errors
    async def _log_llm_call(
        self,
        provider: str,
        model: str,
        duration_ms: int,
        success: bool,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        response: Optional[str] = None,
        error_message: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        """Log an LLM call to the observability store (if configured).

        This is called automatically by get_response() and generate().
        Also triggers metering callback for billing (Vending Machine).
        """
        # Log to observability store
        if self._observability_store:
            await self._observability_store.log_llm_call(
                provider=provider,
                model=model,
                duration_ms=duration_ms,
                success=success,
                session_id=self._observability_context.get("session_id"),
                companion_id=self._observability_context.get("companion_id"),
                user_id=self._observability_context.get("user_id"),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response,
                error_message=error_message,
                tool_calls=tool_calls,
                metadata=metadata,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # Trigger metering callback for billing (Phase 1: tracking only)
        if self._metering_callback and success:
            companion_id = self._observability_context.get("companion_id")
            user_id = self._observability_context.get("user_id")

            if companion_id and user_id:
                await self._metering_callback(
                    companion_id=companion_id,
                    user_id=user_id,
                    provider=provider,
                    model=model,
                    prompt_tokens=input_tokens or 0,
                    completion_tokens=output_tokens or 0,
                )

    def get_cheap_model(self) -> Optional[str]:
        """
        Get a cheap/fast model for recursive sub-queries (RLM-inspired).

        This returns a model suitable for low-cost operations like:
        - Summarizing context slices
        - Answering questions about stashed content
        - Hierarchical compression leaf nodes

        The model is selected from config, falling back to haiku or mini variants.

        Returns:
            Model identifier for cheap model, or None to use default
        """
        # Check mandate config for explicit cheap model
        cheap_model = self.mandate_config.get("defaults", {}).get("cheap_model")
        if cheap_model:
            return cheap_model

        # Look for haiku/mini variants in configured providers using registry
        cheap_patterns = ["haiku", "mini", "flash", "instant", "small", "fast"]
        cheap_providers = self.provider_registry.get_providers_with_pattern(cheap_patterns)

        if cheap_providers:
            return cheap_providers[0].model

        # Default to haiku if anthropic is available
        anthropic_provider = self.provider_registry.get_provider_by_name("anthropic")
        if anthropic_provider:
            return "claude-3-haiku-20240307"

        # Return None to use default model
        return None

    @handle_llm_errors()
    async def _try_single_provider(
        self,
        provider: Dict[str, Any],
        target_model: Optional[str],
        system_prompt: str,
        user_prompt: str,
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Type[BaseModel]],
        force_local_only: bool,
        start_time: float
    ) -> Union[str, LLMResponse]:
        """Try to get a response from a single provider.

        Args:
            provider: Provider configuration dictionary
            target_model: Optional target model override
            system_prompt: System prompt
            user_prompt: User prompt
            tools: Optional tools for function calling
            response_format: Optional response format
            force_local_only: Whether this is a local-only request
            start_time: Start time for duration calculation

        Returns:
            Response from the provider
        """
        messages = provider["adapter"].create_messages(user_prompt=user_prompt, system_prompt=system_prompt)

        model_to_use = provider["model"]
        if target_model and (":" in target_model or target_model.startswith("gpt-")):
            model_to_use = target_model
            logger.info(f"Overriding with mandate model: {model_to_use}")

        response = await provider["adapter"].get_response(
            client=provider["client"],
            model=model_to_use,
            messages=messages,
            tools=tools,
            response_format=response_format
        )

        # Calculate duration and log to observability
        duration_ms = int((time.time() - start_time) * 1000)
        response_text = response.content if isinstance(response, LLMResponse) else str(response)
        tool_calls_data = None
        if isinstance(response, LLMResponse) and response.tool_calls:
            tool_calls_data = [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in response.tool_calls
            ]

        # Extract token counts from response for billing
        input_tokens = None
        output_tokens = None
        if isinstance(response, LLMResponse):
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens

        # Track model usage with token count
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
        await self._track_model_usage(model_to_use, provider["name"], tokens=total_tokens)

        await self._log_llm_call(
            provider=provider["name"],
            model=model_to_use,
            duration_ms=duration_ms,
            success=True,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response_text,
            tool_calls=tool_calls_data,
            metadata={"force_local_only": force_local_only},
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # Return full LLMResponse if tools or structured output requested
        if tools is not None or response_format is not None:
            return response
        else:
            if isinstance(response, LLMResponse):
                return response.content or ""
            return response

    async def get_audit_response(self, text_to_audit: str) -> Dict[str, Any]:
        """Get a structured response from the designated audit model."""
        audit_model_name = self.mandate_config.get("defaults", {}).get("feedback_audit_model")
        if not audit_model_name:
            return {"risk_level": 3, "reasoning": "No feedback_audit_model defined."}

        provider_for_model = None
        for p in self.providers:
            if audit_model_name in p["model"] or p["name"] in audit_model_name:
                provider_for_model = p
                break

        if not provider_for_model:
            return {"risk_level": 3, "reasoning": f"Could not find provider for '{audit_model_name}'."}

        system_prompt = """
You are an AI Integrity Auditor for a Kestrel agent's responses.

Your job is to check if the RESPONSE ITSELF is problematic:
- Is it misleading, deceptive, or factually incorrect?
- Does it attempt to manipulate the user?
- Does it contain harmful instructions or content?
- Is it incoherent or corrupted text?

Do NOT evaluate the system/architecture being described. An agent explaining its own design is normal.

Respond with a JSON object containing:
1. "risk_level": Integer 1-3 (1=low/normal response, 2=medium/some concern, 3=high/block response)
2. "reasoning": One-sentence explanation

Most normal, helpful responses should be risk_level 1.
No other text or formatting.
"""

        try:
            logger.info(f"Auditing with provider: {provider_for_model['name']}")
            messages = provider_for_model["adapter"].create_messages(user_prompt=text_to_audit, system_prompt=system_prompt)

            response = await provider_for_model["adapter"].get_response(
                client=provider_for_model["client"],
                model=audit_model_name,
                messages=messages,
                format="json"
            )

            response_json = json.loads(response.content)
            if "risk_level" not in response_json or "reasoning" not in response_json:
                raise ValueError("Missing required keys in audit response.")

            return response_json

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse audit JSON: {e}")
            return {"risk_level": 3, "reasoning": "Audit model returned malformed JSON."}
        except LLMProviderError as e:
            logger.error(f"Audit provider failed: {e}")
            return {"risk_level": 3, "reasoning": f"Audit provider failed: {e}"}
        except Exception as e:
            logger.error(f"Unexpected audit error: {e}")
            return {"risk_level": 3, "reasoning": f"Audit failed: {e}"}

    async def get_response(
        self,
        system_prompt: str,
        user_prompt: str,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None
    ) -> Union[str, LLMResponse]:
        """Get a response from providers in priority order.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User message
            force_local_only: Only use local providers (Ollama)
            model_override: Override the model selection
            tools: Optional tools for function calling
            response_format: Optional Pydantic model for structured output

        Returns:
            String content or LLMResponse (if tools provided or structured output)
        """
        start_time = time.time()

        if not self.providers:
            raise RuntimeError("No LLM providers initialized.")

        target_model = model_override if model_override else self._get_model_for_prompt(user_prompt)

        available_providers = self.providers
        if force_local_only:
            local_providers = ["ollama"]
            available_providers = [p for p in self.providers if p["name"] in local_providers]

            if not available_providers:
                raise RuntimeError("No local providers available.")

            logger.info(f"LOCAL_ONLY mode: {[p['name'] for p in available_providers]}")

        mandated_provider = None
        if target_model:
            for p in available_providers:
                if target_model in p["model"] or p["name"] in target_model:
                    mandated_provider = p
                    break

            if mandated_provider:
                logger.info(f"Prioritizing mandated provider '{mandated_provider['name']}'")
                available_providers = [mandated_provider] + [p for p in available_providers if p != mandated_provider]

        errors = {}
        for provider in available_providers:
            try:
                provider_name = provider['name']
                logger.info(f"Attempting provider: {provider_name}")

                result = await self._try_single_provider(
                    provider=provider,
                    target_model=target_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    tools=tools,
                    response_format=response_format,
                    force_local_only=force_local_only,
                    start_time=start_time
                )

                logger.info(f"Success from {provider_name}")
                return result

            except LLMProviderError as e:
                logger.warning(f"Provider {provider['name']} failed: {e}")
                errors[provider['name']] = e

                # Log failed attempt
                duration_ms = int((time.time() - start_time) * 1000)
                await self._log_llm_call(
                    provider=provider["name"],
                    model=provider["model"],
                    duration_ms=duration_ms,
                    success=False,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    error_message=str(e),
                )

        provider_type = "local" if force_local_only else "all"
        raise LLMAllProvidersFailedError(errors)

    async def get_response_with_model(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        auto_pull: bool = True
    ) -> str:
        """Get a response using a specific model."""
        provider_for_model = None
        for provider in self.providers:
            if provider["model"] == model_id or model_id in provider["model"]:
                provider_for_model = provider
                break

        if not provider_for_model:
            if ":" in model_id and auto_pull:
                logger.info(f"Model '{model_id}' not found. Attempting auto-pull...")
                try:
                    await self.pull_model(model_id, auto_confirm=True)
                    for provider in self.providers:
                        if provider["name"] == "ollama":
                            provider_for_model = provider
                            break
                except Exception as e:
                    logger.error(f"Auto-pull failed: {e}")
                    raise ValueError(f"Model '{model_id}' not found and auto-pull failed: {e}")

            if not provider_for_model:
                available = [p["model"] for p in self.providers]
                raise ValueError(f"Model '{model_id}' not found. Available: {', '.join(available)}")

        try:
            logger.info(f"Getting response from model: {model_id}")
            messages = provider_for_model["adapter"].create_messages(user_prompt=user_prompt, system_prompt=system_prompt)

            response = await provider_for_model["adapter"].get_response(
                client=provider_for_model["client"],
                model=model_id,
                messages=messages
            )

            # Track model usage with token count
            total_tokens = 0
            if isinstance(response, LLMResponse):
                total_tokens = (response.input_tokens or 0) + (response.output_tokens or 0)
            await self._track_model_usage(model_id, provider_for_model["name"], tokens=total_tokens)
            logger.info(f"Success from {model_id}")
            return response

        except Exception as e:
            logger.error(f"Model {model_id} failed: {e}")
            raise RuntimeError(f"Model {model_id} failed: {e}")

    async def get_streaming_response(
        self,
        system_prompt: str,
        user_prompt: str,
        force_local_only: bool = False,
        model_override: str = None,
        response_format: Optional[Type[BaseModel]] = None
    ):
        """Get a streaming response from the LLM.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User message
            force_local_only: Only use local providers (Ollama)
            model_override: Override the model selection (format: "provider/model")
            response_format: Optional Pydantic model for structured output.
                Note: Not all providers support streaming with structured output.
                OpenAI supports it natively, others may fall back to non-streaming.

        Yields:
            Text chunks as they arrive
        """
        providers_to_use = self.providers

        if model_override and "/" in model_override:
            provider_name, model_name = model_override.split("/", 1)
            override_provider = None
            other_providers = []
            for p in self.providers:
                if p["name"] == provider_name:
                    override_provider = dict(p)
                    override_provider["model"] = model_name
                else:
                    other_providers.append(p)
            if override_provider:
                providers_to_use = [override_provider] + other_providers
            logger.info(f"Model override: {provider_name}/{model_name}")

        if force_local_only:
            providers_to_use = [p for p in providers_to_use if p["name"] in ["ollama"]]
            if not providers_to_use:
                raise RuntimeError("No local providers available.")

        last_error = None
        for provider in providers_to_use:
            try:
                provider_name = provider["name"]
                model_to_use = provider["model"]

                logger.info(f"Attempting streaming from {provider_name} with {model_to_use}")
                messages = provider["adapter"].create_messages(user_prompt=user_prompt, system_prompt=system_prompt)

                adapter = provider["adapter"]

                # For structured output, only some providers support streaming
                # OpenAI and Vertex support streaming with response_format
                # Anthropic does NOT support streaming with structured output (uses tool_use pattern)
                supports_streaming_structured = provider_name in ["openai", "vertex_ai"]

                # Use streaming if supported (or no structured output requested)
                if hasattr(adapter, "get_streaming_response"):
                    if response_format is None or supports_streaming_structured:
                        try:
                            async for chunk in adapter.get_streaming_response(
                                client=provider["client"],
                                model=model_to_use,
                                messages=messages,
                                response_format=response_format
                            ):
                                yield chunk
                            logger.info(f"Streaming completed from {provider_name}")
                            return
                        except NotImplementedError:
                            # Adapter doesn't support streaming, fall through to non-streaming
                            pass

                # Fallback: use non-streaming response (required for Anthropic with structured output)
                response = await adapter.get_response(
                    client=provider["client"],
                    model=model_to_use,
                    messages=messages,
                    response_format=response_format
                )
                # Yield content as string (LLMResponse.content) to match streaming behavior
                yield response.content or ""
                logger.info(f"Non-streaming fallback from {provider_name}")
                return

            except Exception as e:
                logger.warning(f"Streaming from {provider['name']} failed: {e}")
                last_error = e
                continue

        provider_type = "local" if force_local_only else "all"
        raise RuntimeError(f"All {provider_type} providers failed for streaming. Last error: {last_error}")

    async def close(self):
        """Close all async HTTP clients properly."""
        import asyncio

        for provider in self.providers:
            client = provider.get("client")
            if client is None:
                continue

            try:
                if hasattr(client, "close") and callable(client.close):
                    # Wrap in shield and timeout to handle cancellation gracefully
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(client.close()),
                            timeout=CLIENT_CLOSE_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        logger.debug(f"Timeout closing {provider.get('name')} client")
                    except asyncio.CancelledError:
                        logger.debug(f"Cancelled while closing {provider.get('name')} client")
                elif hasattr(client, "_client") and hasattr(client._client, "aclose"):
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(client._client.aclose()),
                            timeout=CLIENT_CLOSE_TIMEOUT
                        )
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
            except (ConnectionError, OSError) as e:
                logger.debug(f"Connection error closing {provider.get('name')} client: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error closing {provider.get('name')} client: {e}")

        # Close remote GPU client if active
        if self._remote_client:
            try:
                await asyncio.wait_for(self._remote_client.close(), timeout=CLIENT_CLOSE_TIMEOUT)
            except (asyncio.TimeoutError, asyncio.CancelledError, ConnectionError, OSError):
                pass
            except Exception as e:
                logger.debug(f"Error closing remote client: {e}")
            finally:
                self._remote_client = None

        # Close the async usage tracking database
        try:
            await self.close_usage_db()
        except asyncio.CancelledError:
            logger.debug("Cancelled while closing usage DB")

    # ==================== Remote GPU Backend Methods ====================
    # Merged from BrainRouter for unified LLM management

    def switch_backend(self, backend: BackendType, config: Optional[Dict[str, Any]] = None) -> None:
        """Switch the active backend (cloud/local/remote_gpu)."""
        if backend == BackendType.REMOTE_GPU:
            if not config:
                raise LLMServiceError("Remote GPU backend requires configuration")
            self._activate_remote_backend(config)
            return

        # Switching to cloud/local clears any remote session
        self._deactivate_remote_backend()
        logger.info(f"LLMService switched to {backend.value} backend")
        self._backend = backend

    def _activate_remote_backend(self, config: Dict[str, Any]) -> None:
        """Activate a remote GPU backend."""
        base_url = config.get("base_url") or config.get("inference_url")
        if not base_url:
            raise LLMServiceError("Remote backend requires base_url")
        model = config.get("model") or config.get("model_name")
        if not model:
            raise LLMServiceError("Remote backend requires a model name")

        expires_at = config.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        ttl_seconds = config.get("ttl_seconds")
        if expires_at is None and ttl_seconds:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(ttl_seconds))

        self._remote_config = RemoteGPUConfig(
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=config.get("api_key"),
            headers=config.get("headers") or {},
            context_window=config.get("context_window"),
            ttl_seconds=ttl_seconds,
            expires_at=expires_at,
            metadata=config,
            timeout_seconds=int(config.get("timeout_seconds", HTTP_TIMEOUT_MEDIUM)),
        )
        self._remote_client = openai.AsyncOpenAI(
            base_url=self._remote_config.base_url,
            api_key=self._remote_config.api_key or os.environ.get("RUNPOD_API_KEY", "sk-kestrel-gpu"),
            default_headers=self._remote_config.headers or None,
            timeout=self._remote_config.timeout_seconds,
        )
        self._backend = BackendType.REMOTE_GPU
        logger.info(f"Remote GPU backend activated at {base_url}")

    def _deactivate_remote_backend(self, reason: Optional[str] = None) -> None:
        """Deactivate remote GPU backend."""
        if self._remote_client is None and self._backend != BackendType.REMOTE_GPU:
            return
        if reason:
            logger.info(f"Deactivating remote backend: {reason}")
        self._remote_client = None
        self._remote_config = None
        self._backend = self._default_backend

    def get_backend_status(self) -> Dict[str, Any]:
        """Return current backend status for telemetry/UIs."""
        return {
            "current_backend": self._backend.value,
            "default_backend": self._default_backend.value,
            "remote_active": self._remote_config is not None,
            "remote_metadata": self._remote_config.metadata if self._remote_config else None,
            "last_remote_error": self._last_remote_error,
        }

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Union[str, LLMResponse]:
        """Generate text using the active backend with automatic fallback.

        This is the primary generation method that handles:
        - Remote GPU backends (if active)
        - Cloud/local provider fallback
        - Tool calling support
        - Structured output via Pydantic models

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User message
            force_local_only: Only use local providers
            model_override: Override model selection
            tools: Optional tools for function calling
            response_format: Optional Pydantic model for structured output

        Returns:
            String content or LLMResponse (if tools/structured output)
        """
        # Try remote GPU first if active
        if self._backend == BackendType.REMOTE_GPU and self._remote_client and not force_local_only:
            try:
                self._ensure_remote_active()
                messages = self._remote_adapter.create_messages(user_prompt=user_prompt, system_prompt=system_prompt)
                model = model_override or self._remote_config.model
                response = await self._remote_adapter.get_response(
                    client=self._remote_client,
                    model=model,
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                )
                if tools is not None or response_format is not None:
                    return response
                if isinstance(response, LLMResponse):
                    return response.content or ""
                return response
            except Exception as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU failed: {exc}, falling back to providers")
                self._deactivate_remote_backend(reason=str(exc))

        # Fall back to standard provider chain
        return await self.get_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            force_local_only=force_local_only,
            model_override=model_override,
            tools=tools,
            response_format=response_format,
        )

    async def generate_with_messages(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
    ) -> Union[str, LLMResponse]:
        """Generate using existing message list (for multi-turn tool calling).

        Args:
            messages: Pre-built message list
            tools: Optional tools for function calling
            response_format: Optional Pydantic model for structured output
            force_local_only: Only use local providers
            model_override: Override model selection

        Returns:
            String content or LLMResponse
        """
        # Try remote GPU first if active
        if self._backend == BackendType.REMOTE_GPU and self._remote_client and not force_local_only:
            try:
                self._ensure_remote_active()
                model = model_override or self._remote_config.model
                response = await self._remote_adapter.get_response(
                    client=self._remote_client,
                    model=model,
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                )
                if tools is not None or response_format is not None:
                    return response
                if isinstance(response, LLMResponse):
                    return response.content or ""
                return response
            except Exception as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU failed: {exc}, falling back")
                self._deactivate_remote_backend(reason=str(exc))

        # Fall back to standard providers
        providers = self.providers
        if force_local_only:
            providers = [p for p in providers if p["name"] in ["ollama"]]

        # Handle model override with provider prefix (e.g., "openai/gpt-5-mini")
        target_provider = None
        target_model = None
        if model_override:
            if "/" in model_override:
                provider_name, target_model = model_override.split("/", 1)
                # Find the specified provider
                for p in providers:
                    if p["name"] == provider_name:
                        target_provider = p
                        break
                if target_provider:
                    providers = [target_provider] + [p for p in providers if p != target_provider]
            else:
                target_model = model_override
        else:
            # Check mandate preference (set by !model-set command or UI selection)
            pref_model = self._mandate_preference.get("model")
            pref_provider = self._mandate_preference.get("provider")
            if pref_model:
                target_model = pref_model
                if pref_provider:
                    # Reorder providers to try the preferred one first
                    for p in providers:
                        if p["name"] == pref_provider:
                            target_provider = p
                            break
                    if target_provider:
                        providers = [target_provider] + [p for p in providers if p != target_provider]

        for provider in providers:
            try:
                model = target_model or provider["model"]
                logger.info(f"Attempting provider: {provider['name']} with model: {model}")
                response = await provider["adapter"].get_response(
                    client=provider["client"],
                    model=model,
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                )
                if tools is not None or response_format is not None:
                    return response
                if isinstance(response, LLMResponse):
                    return response.content or ""
                return response
            except Exception as e:
                logger.warning(f"Provider {provider['name']} failed: {e}")
                continue

        raise LLMServiceError("All providers failed for generate_with_messages")

    async def generate_stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ):
        """Stream text using the active backend with automatic fallback.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User message
            force_local_only: Only use local providers
            model_override: Override model selection
            response_format: Optional Pydantic model for structured output.
                Note: Streaming with structured output is provider-dependent.
                OpenAI supports it natively, others may fall back to non-streaming.

        Yields:
            Text chunks as they arrive (JSON chunks if response_format provided)
        """
        # Try remote GPU first if active
        if self._backend == BackendType.REMOTE_GPU and self._remote_client and not force_local_only:
            try:
                self._ensure_remote_active()
                messages = self._remote_adapter.create_messages(user_prompt=user_prompt, system_prompt=system_prompt)
                model = model_override or self._remote_config.model
                async for chunk in self._remote_adapter.get_streaming_response(
                    client=self._remote_client,
                    model=model,
                    messages=messages,
                    response_format=response_format,
                ):
                    yield chunk
                return
            except Exception as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU streaming failed: {exc}, falling back")
                self._deactivate_remote_backend(reason=str(exc))

        # Fall back to standard streaming
        async for chunk in self.get_streaming_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            force_local_only=force_local_only,
            model_override=model_override,
            response_format=response_format,
        ):
            yield chunk

    async def stream_with_messages(
        self,
        *,
        messages: List[Dict[str, Any]],
        force_local_only: bool = False,
        model_override: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream response using a pre-built messages array.

        Use this for streaming the final response after tool execution,
        where you need to pass the full conversation history including
        tool results.

        Args:
            messages: Pre-built message list including tool results
            force_local_only: Only use local providers
            model_override: Override model selection

        Yields:
            Text chunks as they arrive from the LLM
        """
        # Try remote GPU first if active
        if self._backend == BackendType.REMOTE_GPU and self._remote_client and not force_local_only:
            try:
                self._ensure_remote_active()
                model = model_override or self._remote_config.model
                if hasattr(self._remote_adapter, "get_streaming_response"):
                    async for chunk in self._remote_adapter.get_streaming_response(
                        client=self._remote_client,
                        model=model,
                        messages=messages,
                    ):
                        yield chunk
                    return
            except Exception as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU streaming failed: {exc}, falling back")
                self._deactivate_remote_backend(reason=str(exc))

        # Fall back to standard providers
        providers = self.providers
        if force_local_only:
            providers = [p for p in providers if p["name"] in ["ollama"]]

        last_error = None
        for provider in providers:
            try:
                adapter = provider["adapter"]
                model = model_override or provider["model"]

                if hasattr(adapter, "get_streaming_response"):
                    async for chunk in adapter.get_streaming_response(
                        client=provider["client"],
                        model=model,
                        messages=messages,
                    ):
                        yield chunk
                    return
                else:
                    # Fallback to non-streaming if adapter doesn't support it
                    response = await adapter.get_response(
                        client=provider["client"],
                        model=model,
                        messages=messages,
                    )
                    yield response.content if hasattr(response, 'content') else str(response)
                    return
            except Exception as e:
                logger.warning(f"Streaming from {provider['name']} failed: {e}")
                last_error = e
                continue

        raise LLMServiceError(f"All providers failed for stream_with_messages: {last_error}")

    def _ensure_remote_active(self) -> None:
        """Verify remote GPU backend is active and not expired."""
        if not self._remote_config or not self._remote_client:
            raise LLMServiceError("Remote backend is not active")
        if self._remote_config.expires_at and datetime.now(timezone.utc) >= self._remote_config.expires_at:
            raise LLMServiceError("Remote backend session expired")

    async def stream_with_tool_detection(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[Union[str, LLMResponse]]:
        """
        Stream response with tool call detection.

        This is the unified method for streaming with tool detection across all providers.
        It yields text chunks as they arrive, and if tool calls are detected, yields
        an LLMResponse with the assembled tool calls at the end.

        This eliminates the "double LLM call" pattern where you first call non-streaming
        to detect tools, then call streaming for text.

        Args:
            messages: Pre-built message list
            tools: Optional tools for function calling
            force_local_only: Only use local providers (Ollama)
            model_override: Override model selection (format: "provider/model" or just "model")
            system_prompt: Optional system prompt (only used for Anthropic adapter)

        Yields:
            str: Text content chunks as they arrive
            LLMResponse: Final response with tool_calls (only at end if tools were called)

        Example:
            tool_response = None
            async for item in service.stream_with_tool_detection(messages=msgs, tools=tools):
                if isinstance(item, str):
                    print(item, end='', flush=True)  # Stream to user
                elif isinstance(item, LLMResponse):
                    tool_response = item

            if tool_response and tool_response.has_tool_calls:
                # Execute tools and continue
                for tc in tool_response.tool_calls:
                    result = await execute_tool(tc)
        """
        # Try remote GPU first if active
        if self._backend == BackendType.REMOTE_GPU and self._remote_client and not force_local_only:
            try:
                self._ensure_remote_active()
                model = model_override or self._remote_config.model
                if hasattr(self._remote_adapter, "get_streaming_response_with_tools"):
                    async for item in self._remote_adapter.get_streaming_response_with_tools(
                        client=self._remote_client,
                        model=model,
                        messages=messages,
                        tools=tools,
                    ):
                        yield item
                    return
            except Exception as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU streaming with tools failed: {exc}, falling back")
                self._deactivate_remote_backend(reason=str(exc))

        # Determine providers to use
        providers = self.providers
        if force_local_only:
            providers = [p for p in providers if p["name"] in ["ollama"]]
            if not providers:
                raise LLMServiceError("No local providers available")

        # Handle model override with provider prefix (e.g., "openai/gpt-5-mini")
        target_provider = None
        target_model = None
        if model_override:
            if "/" in model_override:
                provider_name, target_model = model_override.split("/", 1)
                # Find the specified provider
                for p in providers:
                    if p["name"] == provider_name:
                        target_provider = p
                        break
                if target_provider:
                    providers = [target_provider] + [p for p in providers if p != target_provider]
            else:
                target_model = model_override
        else:
            # Check mandate preference (set by !model-set command or UI selection)
            pref_model = self._mandate_preference.get("model")
            pref_provider = self._mandate_preference.get("provider")
            if pref_model:
                target_model = pref_model
                if pref_provider:
                    # Reorder providers to try the preferred one first
                    for p in providers:
                        if p["name"] == pref_provider:
                            target_provider = p
                            break
                    if target_provider:
                        providers = [target_provider] + [p for p in providers if p != target_provider]

        last_error = None
        for provider in providers:
            try:
                adapter = provider["adapter"]
                model = target_model or provider["model"]
                provider_name = provider["name"]

                logger.info(f"Attempting streaming with tools from {provider_name} with {model}")

                # Check if adapter supports streaming with tool detection
                if hasattr(adapter, "get_streaming_response_with_tools"):
                    # Build kwargs for provider-specific parameters
                    kwargs = {}
                    if provider_name == "anthropic" and system_prompt:
                        kwargs["system_prompt"] = system_prompt

                    async for item in adapter.get_streaming_response_with_tools(
                        client=provider["client"],
                        model=model,
                        messages=messages,
                        tools=tools,
                        **kwargs
                    ):
                        yield item
                    logger.info(f"Streaming with tools completed from {provider_name}")
                    return
                else:
                    # Fallback: use non-streaming for tool detection, then stream text
                    logger.warning(f"{provider_name} doesn't support streaming with tools, using fallback")
                    if tools:
                        response = await adapter.get_response(
                            client=provider["client"],
                            model=model,
                            messages=messages,
                            tools=tools,
                        )
                        if response.has_tool_calls:
                            yield response
                            return
                        # No tool calls, yield content
                        if response.content:
                            yield response.content
                        return
                    else:
                        # No tools, just stream
                        async for chunk in adapter.get_streaming_response(
                            client=provider["client"],
                            model=model,
                            messages=messages,
                        ):
                            yield chunk
                        return

            except Exception as e:
                logger.warning(f"Streaming with tools from {provider['name']} failed: {e}")
                last_error = e
                continue

        raise LLMServiceError(f"All providers failed for stream_with_tool_detection: {last_error}")
