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

        Returns compact summaries to avoid context blowout (765+ models).
        """
        try:
            models = await self.llm_service.discover_all_models(use_cache=use_cache)
            # Return compact list: only featured/non-hidden models with essential fields
            compact = []
            for m in models:
                if m.is_hidden:
                    continue
                compact.append({
                    "id": m.id,
                    "provider": m.provider,
                    "category": m.category.value if hasattr(m.category, 'value') else str(m.category),
                    "featured": m.is_featured,
                })
            return compact
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
        description="Report the currently active AI model. Read-only; takes no arguments.",
        category=ToolCategory.MODEL_MANAGEMENT,
        command_prefix="!model"
    )
    async def get_current_model(self) -> Dict[str, Any]:
        """Report the currently active ``{vendor, model, route}``.

        Pure read — never mutates mandate state. The tool used to accept an
        optional ``model`` argument and delegate to ``set_model`` when one was
        given ("dual-purpose"), but the LLM could and did invoke this tool
        with a hallucinated model argument when asked to *report* its model —
        silently rewriting the mandate to a vendor-less bare id and sending
        the next request into a broadcast cascade across every provider.
        Setting is now only reachable via the separate ``set_model`` tool.
        """
        from kestrel_sovereign.llm.service import resolve_active_model_selection
        selection = resolve_active_model_selection(self.llm_service)
        model_str = selection["model"]
        vendor = selection.get("vendor")
        route = selection.get("route")
        model_name = selection.get("model_name")
        return {
            "current_model": model_str,
            "vendor": vendor,
            "route": route,
            "model_name": model_name,
            "message": (
                f"Current model: {model_str}\n\n"
                "Use `!model-set <vendor[:route]> <model>` to change. "
                "Use `!model-list` to list available models."
            ),
        }

    @tool(
        name="set_model",
        description="Set the active AI model for conversations.",
        category=ToolCategory.MODEL_MANAGEMENT,
        command_prefix="!model-set"
    )
    async def set_model(self, vendor_or_model: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Set the active ``{vendor, model, route?}`` with UI sync support.

        Supports two invocation styles:
        - Two args: ``!model-set <vendor[:route]> <model>`` (UI dropdowns)
        - One arg:  ``!model-set <vendor[:route]/model>`` or ``!model-set <model>``

        Args:
            vendor_or_model: Vendor (``"openai"``), composite ``"vendor:route"``
                (``"anthropic:plan"``), or model ID if single arg.
            model: Model ID. If omitted, first arg is parsed as ``vendor/model``
                or ``vendor:route/model`` or a bare model.

        Returns:
            Dict with success status and MODEL_CHANGED marker for UI sync.
        """
        import json

        # Parse vendor/route/model from args.
        vendor: Optional[str] = None
        route: Optional[str] = None
        if model is not None:
            # Two-arg: first is vendor or "vendor:route".
            left = vendor_or_model
            if ":" in left:
                vendor, route = left.split(":", 1)
            else:
                vendor = left
            model_name = model
        else:
            # One-arg.
            model_id = vendor_or_model
            is_openrouter_model = await self._is_openrouter_model(model_id)
            if is_openrouter_model:
                vendor = "openrouter"
                model_name = model_id  # Keep full vendor/model ID as-is.
            elif "/" in model_id:
                left, model_name = model_id.split("/", 1)
                if ":" in left:
                    vendor, route = left.split(":", 1)
                else:
                    vendor = left
            else:
                model_name = model_id

        try:
            # Context safety check: use the same pruning logic the actual LLM
            # call path uses. If format_conversation_history can fit the history
            # into the new model's history budget (after per-message cap and
            # pruning), the switch is safe.
            if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'storage'):
                history = await self.agent.storage.get_conversation_history(limit=50)

                # Use the agent's context_builder if available; otherwise
                # construct a temporary one — it just needs the token counter.
                ctx_builder = getattr(self.agent, 'context_builder', None)
                if ctx_builder is None:
                    from kestrel_sovereign.agent.context_builder import ContextBuilder
                    ctx_builder = ContextBuilder(storage=self.agent.storage)

                est = ctx_builder.estimate_effective_history_tokens(history, model_name)

                # The switch fails only if, after pruning, the effective history
                # exceeds the history budget by more than 5% slack (absorbs
                # truncation marker overhead and per-message rounding).
                overflow_tolerance = max(int(est['history_budget'] * 0.05), 256)
                if est['effective_tokens'] > est['history_budget'] + overflow_tolerance:
                    overflow = est['effective_tokens'] - est['history_budget']
                    utilization = (est['effective_tokens'] / est['history_budget'] * 100)
                    return {
                        "success": False,
                        "error": "context_overflow",
                        "message": (
                            f"⚠️ Cannot switch to {model_name}: context too small even after pruning.\n\n"
                            f"Effective history after pruning: {est['effective_tokens']:,} tokens\n"
                            f"History budget on new model: {est['history_budget']:,} tokens\n"
                            f"Raw history (for reference): {est['raw_tokens']:,} tokens\n"
                            f"Overflow: {overflow:,} tokens ({utilization:.1f}%)\n\n"
                            f"The new model's context window ({est['context_limit']:,}) is too small for this conversation.\n"
                            f"Try a model with a larger context, or run `!compress` to reduce history."
                        )
                    }

            # Record agent consent before applying the change
            consent = self.agent.features.get("ConsentFeature") if hasattr(self.agent, 'features') else None
            if consent:
                try:
                    current_pref = self.llm_service.get_model_preference()
                    current_model = current_pref.get('model', 'unknown')
                    await consent.request_consent(
                        "model_change",
                        {"from": current_model, "to": model_name, "vendor": vendor, "route": route},
                    )
                except Exception:
                    pass  # Never block on consent failure

            # Safe to switch
            self.llm_service.set_model_preference(model_name, vendor, route)
            if vendor and route:
                full_model = f"{vendor}:{route}/{model_name}"
            elif vendor:
                full_model = f"{vendor}/{model_name}"
            else:
                full_model = model_name

            # Return with MODEL_CHANGED marker for UI sync
            sync_data = json.dumps({
                "model": full_model,
                "vendor": vendor,
                "route": route,
                "model_name": model_name,
            })
            return {
                "success": True,
                "model": full_model,
                "vendor": vendor,
                "route": route,
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
            # Primary check: look up in shared model discovery cache
            from kestrel_sovereign.llm.model_cache import get_shared_model_cache
            _cached_models = get_shared_model_cache().get_any()
            if _cached_models:
                for m in _cached_models:
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
        Set the preferred model for the agent from a combined string.

        DEPRECATED: Use ``llm_service.set_model_preference(model, vendor, route)``
        directly. This wrapper exists only for callers that pass a single
        ``"vendor/model"`` or ``"vendor:route/model"`` string.
        """
        import warnings
        warnings.warn(
            "ModelAgent.set_model_preference() is deprecated. "
            "Use llm_service.set_model_preference(model, vendor, route) directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        vendor: Optional[str] = None
        route: Optional[str] = None
        model = model_id
        if "/" in model_id:
            left, model = model_id.split("/", 1)
            if ":" in left:
                vendor, route = left.split(":", 1)
            else:
                vendor = left

        self.llm_service.set_model_preference(model, vendor, route)
        logger.info(f"Model preference set to: {model_id}")
        return f"Model preference set to {model_id}"

    def get_model_preference(self) -> Optional[str]:
        """
        Get the current model preference.

        DEPRECATED: Use llm_service.get_model_preference() directly.
        This returns Optional[str] while the canonical method returns
        Dict[str, Optional[str]] with 'model' and 'provider' keys.
        """
        import warnings
        warnings.warn(
            "ModelAgent.get_model_preference() is deprecated. "
            "Use llm_service.get_model_preference() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        pref = self.llm_service.get_model_preference()
        if pref.get("model"):
            vendor = pref.get("vendor")
            route = pref.get("route")
            model = pref.get("model")
            if vendor and route:
                return f"{vendor}:{route}/{model}"
            if vendor:
                return f"{vendor}/{model}"
            return model
        return None
