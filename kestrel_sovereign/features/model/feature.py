import logging
from typing import List, Dict, Any, Optional
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.llm.model_metadata import ModelInfo
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

logger = logging.getLogger(__name__)


class ModelAgent(Feature):
    """
    Agent responsible for managing LLM models.
    Handles discovery, switching, pulling, and cleanup of models.
    """

    @property
    def tool_description(self) -> str:
        return (
            "Manage LLM models - list available models from all providers, "
            "change the active model, pull new models from Ollama, "
            "check storage usage, and clean up unused models"
        )

    async def initialize(self):
        # LLMService is passed in __init__ by KestrelAgent, but Feature base class expects 'agent'
        # We might need to adjust how we initialize this.
        # For now, let's assume self.agent has llm_service
        if hasattr(self.agent, 'llm_service'):
            self.llm_service = self.agent.llm_service
        else:
            # Fallback or error
            logger.warning("ModelAgent initialized without LLMService on agent")

    @tool(
        name="list_models",
        description="List all available AI models.",
        category=ToolCategory.MODEL_MANAGEMENT,
        command_prefix="!model-list"
    )
    async def list_models(self, use_cache: bool = True) -> List[ModelInfo]:
        """
        List all available models from all providers.

        Returns:
            List of ModelInfo objects describing available models.
        """
        try:
            return await self.llm_service.discover_all_models(use_cache=use_cache)
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise

    @tool(
        name="pull_model",
        description="Download a new AI model (Ollama only).",
        category=ToolCategory.MODEL_MANAGEMENT,
        command_prefix="!model-pull"
    )
    async def pull_model(self, model_name: str, progress_callback=None) -> bool:
        """
        Pull (download) a model (primarily for Ollama).
        """
        try:
            return await self.llm_service.pull_model(
                model_name=model_name,
                auto_confirm=True,
                progress_callback=progress_callback
            )
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            raise

    @tool(
        name="get_model_storage_info",
        description="Get storage usage information for local models.",
        category=ToolCategory.MODEL_MANAGEMENT
    )
    async def get_storage_info(self, use_cache: bool = False) -> Dict[str, Any]:
        """
        Get storage information (primarily for Ollama).
        """
        try:
            return await self.llm_service.get_storage_info(use_cache=use_cache)
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            raise

    @tool(
        name="cleanup_models",
        description="Clean up unused models to free space.",
        category=ToolCategory.MODEL_MANAGEMENT
    )
    async def cleanup_models(self, threshold_days: int = 30, dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean up unused models.
        """
        try:
            return await self.llm_service.cleanup_unused_models(
                threshold_days=threshold_days,
                min_free_space_pct=10,
                dry_run=dry_run
            )
        except Exception as e:
            logger.error(f"Error cleaning up models: {e}")
            raise

    @tool(
        name="get_model_info",
        description="Get detailed information about a specific model.",
        category=ToolCategory.MODEL_MANAGEMENT,
        command_prefix="!model-info"
    )
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        """
        try:
            # Get all models
            models = await self.llm_service.discover_all_models(use_cache=True)

            # Find the requested model (ModelInfo objects have .id attribute)
            model = next((m for m in models if m.id == model_name), None)

            if not model:
                raise ValueError(f"Model not found: {model_name}")

            # Convert to dict for response
            result = model.to_dict()

            # Get usage info if available
            storage = await self.llm_service.get_storage_info(use_cache=True)
            model_info = next((m for m in storage.get('models', []) if m.get('id') == model_name), None)

            if model_info:
                result['last_used'] = model_info.get('last_used', 'never')

            return result
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            raise

    @tool(
        name="get_current_model",
        description="Show the currently active AI model. If a model argument is provided, set that model instead.",
        category=ToolCategory.MODEL_MANAGEMENT,
        command_prefix="!model"
    )
    async def get_current_model(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Show current model or set a new one if model argument is provided.

        Args:
            model: Optional model to set (format: provider/model or just model name)

        Returns:
            Dict with current model info and optional MODEL_CHANGED marker for UI sync
        """
        import json

        # If model argument provided, delegate to set_model
        if model:
            return await self.set_model(model)

        # Get current model from mandate preference
        pref = self.llm_service.get_model_preference()
        model_name = pref.get('model')
        provider = pref.get('provider')

        # If no mandate preference, use the first provider (what actually gets used)
        if not model_name and self.llm_service.providers:
            first_provider = self.llm_service.providers[0]
            provider = first_provider.get('name')
            model_name = first_provider.get('model')

        model_str = f"{provider}/{model_name}" if provider else model_name
        return {
            "current_model": model_str,
            "provider": provider,
            "model_name": model_name,
            "message": f"Current model: {model_str}\n\nUse `!model <provider/model>` to change.\nUse `!model-list` to list available models."
        }

    @tool(
        name="set_model",
        description="Set the active AI model for conversations.",
        category=ToolCategory.MODEL_MANAGEMENT,
        command_prefix="!model-set"
    )
    async def set_model(self, provider_or_model: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Set the active model with UI sync support.

        Supports two formats:
        - Two args: !model-set <provider> <model> (UI dropdowns)
        - One arg: !model-set <model> (direct command input)

        Args:
            provider_or_model: Provider name (e.g., "openrouter") OR model ID if single arg
            model: Model ID (e.g., "google/gemini-3-pro-preview"). If omitted, first arg is model.

        Returns:
            Dict with success status and MODEL_CHANGED marker for UI sync
        """
        import json
        from kestrel_sovereign.agent.token_counter import get_token_counter

        # Determine provider and model based on argument pattern
        if model is not None:
            # Two-arg format: !model-set openrouter google/gemini-3-pro-preview
            # Provider is explicitly specified - use it directly
            provider = provider_or_model
            model_name = model
        else:
            # One-arg format: !model-set google/gemini-3-pro-preview
            model_id = provider_or_model

            # Check if this model belongs to OpenRouter (meta-provider)
            is_openrouter_model = await self._is_openrouter_model(model_id)

            if is_openrouter_model:
                # OpenRouter model - keep full ID, use openrouter as provider
                provider = "openrouter"
                model_name = model_id  # Keep full ID like "google/gemini-3-pro-preview"
            elif '/' in model_id:
                # Direct provider format like "openai/gpt-5" or "ollama/llama3.2"
                provider, model_name = model_id.split('/', 1)
            else:
                # Simple model name without provider
                provider = None
                model_name = model_id

        try:
            # Context safety check: ensure history fits in new model's context
            new_counter = get_token_counter(model_name)
            new_limit = new_counter.get_context_limit()
            new_budget = new_limit - 1024  # Reserve for response

            # Count current history tokens
            if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'storage'):
                history = await self.agent.storage.get_conversation_history(limit=10000)
                total_tokens = sum(new_counter.count(m.get("content", "")) for m in history)

                if total_tokens > new_budget:
                    overflow = total_tokens - new_budget
                    utilization = (total_tokens / new_budget * 100)
                    return {
                        "success": False,
                        "error": "context_overflow",
                        "message": (
                            f"⚠️ Cannot switch to {model_name}: context overflow detected.\n\n"
                            f"Current history: {total_tokens:,} tokens\n"
                            f"New model limit: {new_budget:,} tokens\n"
                            f"Overflow: {overflow:,} tokens ({utilization:.1f}%)\n\n"
                            f"Run `!compress` first to reduce context, then try again."
                        )
                    }

            # Safe to switch
            self.llm_service.set_model_preference(model_name, provider)
            full_model = f"{provider}/{model_name}" if provider else model_name

            # Return with MODEL_CHANGED marker for UI sync
            sync_data = json.dumps({"model": full_model, "provider": provider})
            return {
                "success": True,
                "model": full_model,
                "provider": provider,
                "model_name": model_name,
                "message": f"✓ Model set to: {full_model}\n\nMODEL_CHANGED:{sync_data}"
            }
        except Exception as e:
            logger.error(f"Error setting model to {model}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"❌ Error setting model: {e}"
            }

    async def _is_openrouter_model(self, model_id: str) -> bool:
        """
        Check if a model ID belongs to OpenRouter.

        OpenRouter models have format "vendor/model" (e.g., "google/gemini-3-pro-preview")
        but should be routed through the OpenRouter provider, not the underlying vendor.

        The definitive check is whether the model exists in the cached model discovery
        with provider="openrouter".

        Args:
            model_id: The model identifier to check

        Returns:
            True if this model should be routed through OpenRouter
        """
        if not hasattr(self, 'llm_service') or not self.llm_service:
            return False

        try:
            # Primary check: look up in model discovery cache
            if hasattr(self.llm_service, '_model_cache') and self.llm_service._model_cache:
                for m in self.llm_service._model_cache:
                    if m.id == model_id and m.provider == "openrouter":
                        return True

            # Secondary check: if model has "/" and first part matches a known OpenRouter vendor
            # This handles cases where the model wasn't in cache yet
            if '/' in model_id:
                prefix = model_id.split('/')[0]
                # These vendors ONLY exist on OpenRouter (not as direct providers)
                openrouter_only_vendors = {
                    'deepseek', 'meta-llama', 'mistralai', 'cohere', 'ai21',
                    'perplexity', 'fireworks', 'together', 'groq', 'nvidia',
                    'bytedance-seed', 'minimax', 'z-ai', 'qwen', 'nous',
                    'cognitivecomputations', 'sao10k', 'undi95', 'neversleep',
                    'gryphe', 'teknium', 'koboldai', 'pygmalionai', 'thedrummer'
                }
                if prefix in openrouter_only_vendors:
                    return True

        except Exception as e:
            logger.debug(f"Error checking if {model_id} is OpenRouter model: {e}")

        return False

    def set_model_preference(self, model_id: str) -> str:
        """
        Set the preferred model for the agent (legacy method).
        
        NOTE: This is a legacy method. Prefer using llm_service.set_model_preference(model, provider)
        directly for full control over provider routing.
        """
        # Parse provider from model_id if present
        provider = None
        model = model_id
        if '/' in model_id:
            provider, model = model_id.split('/', 1)
        
        self.llm_service.set_model_preference(model, provider)
        logger.info(f"Model preference set to: {model_id}")
        return f"Model preference set to {model_id}"

    def get_model_preference(self) -> Optional[str]:
        """
        Get the current model preference (legacy method).
        """
        pref = self.llm_service.get_model_preference()
        if pref.get("model"):
            provider = pref.get("provider")
            model = pref.get("model")
            return f"{provider}/{model}" if provider else model
        return None
