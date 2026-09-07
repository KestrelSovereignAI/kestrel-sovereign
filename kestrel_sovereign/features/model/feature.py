import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.security.host_authority import (
    HostAuthorityError,
    require_sovereign_caller,
)

logger = logging.getLogger(__name__)


def _caller_is_sovereign() -> bool:
    """The deletion predicate, asked without raising, for the report path."""
    try:
        require_sovereign_caller("shared local model deletion")
    except HostAuthorityError:
        return False
    return True


@dataclass(frozen=True)
class FleetModelRoster:
    """What a shared-daemon cleanup accounted for, and what it could not."""

    protected: set
    consulted: List[str]
    unconsulted: List[str]


def _configured_agent_names() -> List[str]:
    """Every local agent the host is configured with, loaded or not.

    Resolved exactly as ``kestrel start`` resolves what it launches:
    ``MultiAgentConfig.load`` on the project's ``multi_agent.toml``, which
    falls back to auto-discovering ``agent_data/*`` when the file is absent
    (the file is gitignored; a restored host may have only the directories).
    An earlier version short-circuited on a missing file and reported "no
    fleet" for a fleet the launcher would start — the refusal below could
    then never fire on such a host. A bad file raises, and the tool reports
    it, rather than deleting under a roster it could not read.
    """
    from kestrel_sovereign.multi_agent.config import (
        MULTI_AGENT_CONFIG_FILENAME,
        MultiAgentConfig,
    )
    from kestrel_sovereign.paths import project_dir

    config = MultiAgentConfig.load(project_dir() / MULTI_AGENT_CONFIG_FILENAME)
    return sorted(config.get_local_agents().keys())


class ModelAgent(Feature):
    """
    Agent responsible for managing LLM models.
    Handles discovery, switching, pulling, and cleanup of models.
    """

    # The host roster reader, an instance hook so a test can hand a feature
    # a roster without patching the module for every other test.
    _configured_agent_names = staticmethod(_configured_agent_names)

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
    async def list_models(self, use_cache: bool = True) -> ToolResult:
        """List all available models from all providers.

        Returns compact summaries to avoid context blowout (765+ models).
        """
        try:
            models = await self.llm_service.discover_all_models(use_cache=use_cache)
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return ToolResult.failed(str(e))

        compact: List[Dict[str, Any]] = []
        for m in models:
            if m.is_hidden:
                continue
            compact.append({
                "id": m.id,
                "provider": m.provider,
                "category": m.category.value if hasattr(m.category, 'value') else str(m.category),
                "featured": m.is_featured,
            })
        return ToolResult.ok(
            confirmation=f"Listed {len(compact)} model(s)",
            data={"models": compact, "count": len(compact)},
        )

    @tool(
        name="pull_model",
        description="Download a new AI model (Ollama only).",
        category=ToolCategory.MODEL_MANAGEMENT,
        command_prefix="!model-pull"
    )
    async def pull_model(self, model_name: str, progress_callback=None) -> ToolResult:
        """Pull (download) a model (primarily for Ollama).

        Installs onto the host's shared daemon: the service refuses unless
        the turn carries sovereign authority (#3221). Tool consent (ASK/AUTO)
        is not that authority and cannot promote it.
        """
        try:
            ok = await self.llm_service.pull_model(
                model_name=model_name,
                auto_confirm=True,
                progress_callback=progress_callback
            )
        except HostAuthorityError as e:
            logger.warning(f"Refused shared model pull of {model_name!r}: {e}")
            return ToolResult.failed(
                str(e),
                data={"model_name": model_name, "pulled": False, "authority": "sovereign"},
            )
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return ToolResult.failed(str(e))

        if not ok:
            return ToolResult.failed(
                f"Pull of {model_name!r} did not complete; the underlying "
                "service returned a falsy status. Check the provider logs.",
                data={"model_name": model_name, "pulled": False},
            )
        return ToolResult.ok(
            confirmation=f"Pulled model {model_name!r}",
            data={"model_name": model_name, "pulled": True},
        )

    @tool(
        name="get_model_storage_info",
        description="Get storage usage information for local models.",
        category=ToolCategory.MODEL_MANAGEMENT
    )
    async def get_storage_info(self, use_cache: bool = False) -> ToolResult:
        """Get storage information (primarily for Ollama)."""
        try:
            info = await self.llm_service.get_storage_info(use_cache=use_cache)
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            return ToolResult.failed(str(e))

        return ToolResult.ok(
            confirmation="Retrieved model storage info",
            data=info if isinstance(info, dict) else {"raw": info},
        )

    @tool(
        name="cleanup_models",
        description=(
            "Clean up unused models to free space. DESTRUCTIVE: deletes model "
            "files from disk. Defaults to dry_run=True, which only PREVIEWS what "
            "would be deleted without removing anything — re-run with dry_run=False "
            "to actually delete. Only models unused for threshold_days (default 30) "
            "are eligible, and deletion is gated so at least min_free_space_pct (10%) "
            "free disk is preserved."
        ),
        category=ToolCategory.MODEL_MANAGEMENT
    )
    async def cleanup_models(self, threshold_days: int = 30, dry_run: bool = True) -> ToolResult:
        """Clean up unused models.

        Args:
            threshold_days: Only models unused for at least this many days are eligible for deletion (default: 30).
            dry_run: If True (the default), only preview what would be deleted; nothing is removed. Set False to actually delete.
        """
        try:
            # Inside the try: an unreadable multi_agent.toml is a refusal
            # with the reason, not an exception through the tool wrapper.
            roster = self._fleet_model_roster()
            # The roster is host information (which agents exist here, which
            # are cold). Only a sovereign caller sees it — on the deletion
            # path by asking first, on the report path by asking quietly:
            # a non-sovereign dry run gets a count-free caveat and no names.
            roster_visible = _caller_is_sovereign()
            if not dry_run:
                require_sovereign_caller("shared local model deletion")
                if roster.unconsulted:
                    # A partial roster is not a smaller risk, it is an unknown
                    # one: the models a cold or out-of-process agent is pinned
                    # to are exactly the ones this process cannot see (#3221).
                    names = ", ".join(roster.unconsulted)
                    return ToolResult.failed(
                        "shared local model deletion refused: cannot account for "
                        f"models in use by configured agents this process cannot "
                        f"consult ({names}). Start them on this host, or preview "
                        "with dry_run=True.",
                        data={
                            "dry_run": False,
                            "consulted_agents": roster.consulted,
                            "unconsulted_agents": roster.unconsulted,
                        },
                    )
            result = await self.llm_service.cleanup_unused_models(
                threshold_days=threshold_days,
                min_free_space_pct=10,
                dry_run=dry_run,
                protected_models=roster.protected,
            )
        except HostAuthorityError as e:
            logger.warning(f"Refused shared model cleanup: {e}")
            return ToolResult.failed(
                str(e),
                data={"dry_run": dry_run, "authority": "sovereign"},
            )
        except Exception as e:
            logger.error(f"Error cleaning up models: {e}")
            return ToolResult.failed(str(e))

        data = result if isinstance(result, dict) else {"raw": result}
        # The plan is auditable to the sovereign: which agents' models it
        # accounted for, and which it could not — a real deletion refuses on
        # the latter. Anyone else learns only that the plan is incomplete.
        if roster_visible:
            data["consulted_agents"] = roster.consulted
            data["unconsulted_agents"] = roster.unconsulted

        # Honesty: dry-run is an explicit "did not actually delete"
        # mode. The agent must speak that nothing was actually freed —
        # otherwise an LLM that calls cleanup_models(dry_run=True) and
        # narrates "freed 12GB" would be lying. Surface as PARTIAL with
        # the dry-run caveat so the model cannot omit it.
        if dry_run:
            caveat = (
                "dry_run=True; no models were actually deleted and no "
                "space was freed. Re-run with dry_run=False to apply."
            )
            if roster.unconsulted:
                caveat += (
                    " This plan could not account for every configured agent "
                    "(some cannot be consulted from this process); a real "
                    "deletion will refuse until they are loaded here."
                )
                if roster_visible:
                    caveat += f" Unconsulted: {', '.join(roster.unconsulted)}."
            return ToolResult.partial(
                confirmation="Cleanup planned (dry-run)",
                error=caveat,
                data=data,
            )

        return ToolResult.ok(
            confirmation=(
                f"Cleanup complete (threshold={threshold_days} days)"
            ),
            data=data,
        )

    def _fleet_model_roster(self) -> "FleetModelRoster":
        """Local models any co-hosted agent still needs, and who was asked.

        The Ollama daemon is one per host. Before #3221 cleanup protected
        only the models the *calling* agent's service named, so one agent
        could delete a model another was pinned to. Every agent **loaded in
        this process** contributes its own service's protected set. The
        host's configured roster (``multi_agent.toml``) is the reference:
        an agent configured but not loaded here — ``autostart = false`` and
        still cold, or running in its own process under ``kestrel start
        <name>`` — cannot be consulted, and its pins are exactly the models
        this process cannot see. Those names are reported, and a real
        deletion refuses while any exist. The report (dry run) and the
        deletion use the same set, so the preview is the plan.
        """
        own_name = getattr(self.agent, "agent_name", None)
        services = {own_name or "<self>": self.llm_service}
        manager = getattr(self.agent, "_agent_manager", None) or getattr(
            self.agent, "agent_manager", None
        )
        list_agents = getattr(manager, "list_agents", None)
        if callable(list_agents):
            for name, peer in list(list_agents().items()):
                service = getattr(peer, "llm_service", None)
                if service is not None:
                    services[name] = service
        protected: set = set()
        for service in services.values():
            locally = getattr(service, "locally_protected_models", None)
            if callable(locally):
                protected |= set(locally())
        consulted = sorted(str(name) for name in services)
        unconsulted = sorted(
            name for name in self._configured_agent_names() if name not in services
        )
        return FleetModelRoster(
            protected=protected, consulted=consulted, unconsulted=unconsulted
        )

    @tool(
        name="get_model_info",
        description="Get detailed information about a specific model.",
        category=ToolCategory.MODEL_MANAGEMENT,
        command_prefix="!model-info"
    )
    async def get_model_info(self, model_name: str) -> ToolResult:
        """Get detailed information about a specific model."""
        try:
            models = await self.llm_service.discover_all_models(use_cache=True)
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return ToolResult.failed(str(e))

        model = next((m for m in models if m.id == model_name), None)
        if not model:
            return ToolResult.failed(
                f"Model not found: {model_name}",
                data={"model_name": model_name},
            )

        result = model.to_dict()

        try:
            storage = await self.llm_service.get_storage_info(use_cache=True)
            model_info = next(
                (m for m in storage.get('models', []) if m.get('id') == model_name),
                None,
            )
            if model_info:
                result['last_used'] = model_info.get('last_used', 'never')
        except Exception as e:
            # Storage lookup is enrichment, not load-bearing — log but
            # don't fail the entire info call.
            logger.debug(f"storage info enrichment skipped for {model_name}: {e}")

        # Wrap under a "model" key so the structural-payload heuristic
        # in command_handler renders the details. A flat scalar dict
        # would hide the model details from `!model-info` users.
        return ToolResult.ok(
            confirmation=f"Model {model_name!r} info retrieved",
            data={"model": result},
        )

    @tool(
        name="get_current_model",
        description="Report the currently active AI model. Read-only; takes no arguments.",
        category=ToolCategory.MODEL_MANAGEMENT,
        command_prefix="!model"
    )
    async def get_current_model(self) -> ToolResult:
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
        return ToolResult.ok(
            confirmation=f"Current model: {model_str}",
            data={
                "current_model": model_str,
                "vendor": vendor,
                "route": route,
                "model_name": model_name,
                "message": (
                    f"Current model: {model_str}\n\n"
                    "Use `!model-set <vendor[:route]> <model>` to change. "
                    "Use `!model-list` to list available models."
                ),
            },
        )

    @tool(
        name="set_model",
        description=(
            "Set the active AI model for conversations. Accepts a vendor and "
            "model, e.g. set_model('openai', 'gpt-5-mini'), or the "
            "'vendor:route/model' micro-syntax in a single arg, e.g. "
            "set_model('anthropic:plan/claude-opus-4-7'). The vendor, route, "
            "and model must be real — an unknown triple is rejected, not "
            "silently applied. Call list_models first to discover valid "
            "vendor/route/model values."
        ),
        category=ToolCategory.MODEL_MANAGEMENT,
        command_prefix="!model-set"
    )
    async def set_model(self, vendor_or_model: str, model: Optional[str] = None) -> ToolResult:
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
        """
        import json

        # Parse vendor/route/model from args.
        vendor: Optional[str] = None
        route: Optional[str] = None
        if model is not None:
            left = vendor_or_model
            if ":" in left:
                vendor, route = left.split(":", 1)
            else:
                vendor = left
            model_name = model
        else:
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
            # Context safety check — same pruning logic the actual LLM
            # call path uses.
            if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'storage'):
                history = await self.agent.storage.get_conversation_history(limit=50)

                ctx_builder = getattr(self.agent, 'context_builder', None)
                if ctx_builder is None:
                    from kestrel_sovereign.agent.context_builder import ContextBuilder
                    ctx_builder = ContextBuilder(storage=self.agent.storage)

                est = ctx_builder.estimate_effective_history_tokens(history, model_name)

                # The switch fails only if, after pruning, the effective history
                # exceeds the history budget by more than 5% slack.
                overflow_tolerance = max(int(est['history_budget'] * 0.05), 256)
                if est['effective_tokens'] > est['history_budget'] + overflow_tolerance:
                    overflow = est['effective_tokens'] - est['history_budget']
                    utilization = (est['effective_tokens'] / est['history_budget'] * 100)
                    overflow_message = (
                        f"context_overflow: cannot switch to {model_name}: "
                        f"context too small even after pruning. "
                        f"Effective history {est['effective_tokens']:,} tok > "
                        f"budget {est['history_budget']:,} tok "
                        f"(overflow {overflow:,} tok, {utilization:.1f}%). "
                        f"The new model's context window ({est['context_limit']:,}) "
                        "is too small for this conversation. Try a model with a "
                        "larger context, or run `!compact` to reduce history."
                    )
                    return ToolResult.failed(
                        overflow_message,
                        data={
                            "success": False,
                            "error": "context_overflow",
                            "model_name": model_name,
                            "vendor": vendor,
                            "route": route,
                            "effective_tokens": est['effective_tokens'],
                            "history_budget": est['history_budget'],
                            "raw_tokens": est['raw_tokens'],
                            "overflow_tokens": overflow,
                            "context_limit": est['context_limit'],
                        },
                    )

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
                except Exception as e:
                    logger.warning(f"consent recording failed (non-blocking): {e}")

            # Safe to switch
            self.llm_service.set_model_preference(model_name, vendor, route)
            if vendor and route:
                full_model = f"{vendor}:{route}/{model_name}"
            elif vendor:
                full_model = f"{vendor}/{model_name}"
            else:
                full_model = model_name

            sync_data = json.dumps({
                "model": full_model,
                "vendor": vendor,
                "route": route,
                "model_name": model_name,
            })
            message = f"✓ Model set to: {full_model}\n\nMODEL_CHANGED:{sync_data}"
            data = {
                "success": True,
                "model": full_model,
                "vendor": vendor,
                "route": route,
                "model_name": model_name,
                "message": message,
            }
            # MODEL_CHANGED marker MUST appear in the rendered text so
            # the WebUI's sync parser can pick it up. The envelope
            # formatter renders ``confirmation`` verbatim and drops
            # scalar-only ``data`` (per PR #1093 read-payload heuristic),
            # so the marker has to live in the confirmation itself.
            return ToolResult.ok(confirmation=message, data=data)
        except Exception as e:
            logger.error(f"Error setting model to {model}: {e}")
            return ToolResult.failed(
                str(e),
                data={"success": False, "message": f"❌ Error setting model: {e}"},
            )

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
