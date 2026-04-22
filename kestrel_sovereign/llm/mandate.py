"""Model mandate management for LLM Service."""
import logging
from typing import Dict, Any, Optional, List

from kestrel_sovereign.config import load_config

logger = logging.getLogger(__name__)


class ModelMandateMixin:
    """Mixin class providing model mandate methods for LLMService."""

    def _resolve_model_selector(
        self,
        selector: Optional[str],
        providers: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Optional[str]]:
        """Resolve a mandate selector to a ``(provider-selector, model)`` pair.

        The returned ``provider`` may be a vendor (``"anthropic"``) or a
        composite route key (``"anthropic:plan"``). Callers downstream
        feed it to ``_filter_providers_by_selector``.

        Supported selectors:
            - vendor: ``anthropic``
            - vendor:route: ``anthropic:plan``
            - vendor/model: ``anthropic/claude-sonnet-4-6``
            - vendor:route/model: ``anthropic:plan/claude-sonnet-4-6``
            - bare model id: ``gpt-5-mini``
            - alias: ``cheap`` (defers to ``get_cheap_model``)
        """
        providers = providers or self.providers

        if not selector:
            return {"selector": None, "provider": None, "model": None}

        selector = selector.strip()
        if not selector or selector == "auto":
            return {"selector": None, "provider": None, "model": None}

        if selector == "cheap" and hasattr(self, "get_cheap_model"):
            cheap_model = self.get_cheap_model()
            if not cheap_model or cheap_model == selector:
                return {"selector": None, "provider": None, "model": None}
            return self._resolve_model_selector(cheap_model, providers=providers)

        if "/" in selector:
            left, model_name = selector.split("/", 1)
            return {
                "selector": selector,
                "provider": left,  # vendor or "vendor:route"
                "model": model_name,
            }

        # Vendor-only or composite-route-only match (no model supplied).
        for provider in providers:
            if provider.get("vendor") == selector or provider.get("name") == selector:
                model_name = provider.get("model")
                match_key = selector  # pass back what was asked for; caller filters
                normalized = f"{match_key}/{model_name}" if model_name else match_key
                return {
                    "selector": normalized,
                    "provider": match_key,
                    "model": model_name,
                }

        # Bare model id — find a route whose default model matches.
        for provider in providers:
            if provider.get("model") == selector:
                vendor = provider.get("vendor") or provider.get("name")
                normalized = f"{vendor}/{selector}" if vendor else selector
                return {
                    "selector": normalized,
                    "provider": vendor,
                    "model": selector,
                }

        return {"selector": selector, "provider": None, "model": selector}

    def _get_default_mandate_selector(self) -> Optional[str]:
        """Return the default configured selector, if any."""
        return self.mandate_config.get("defaults", {}).get("preferred") or None

    def _is_banned_selector(self, selector: Optional[str]) -> bool:
        """Return True when selector resolves to a banned provider or model."""
        banned = self.mandate_config.get("defaults", {}).get("banned", [])
        if not selector:
            return False

        resolved = self._resolve_model_selector(selector)
        candidates = {
            selector,
            resolved.get("selector"),
            resolved.get("provider"),
            resolved.get("model"),
        }
        return any(item and item in candidates for item in banned)

    def get_current_mandate(self) -> Dict[str, Any]:
        """Get the current model mandate configuration."""
        preference_model = self._mandate_preference.get("model")
        preference_vendor = self._mandate_preference.get("vendor")
        preference_route = self._mandate_preference.get("route")

        if preference_model is None:
            resolved_default = self._resolve_model_selector(self._get_default_mandate_selector())
            preference_model = resolved_default.get("model")
            # resolved.get("provider") may be vendor or vendor:route — keep the
            # vendor slot populated from it, and try to split route if composite.
            provider_field = resolved_default.get("provider")
            if provider_field and ":" in provider_field:
                preference_vendor, preference_route = provider_field.split(":", 1)
            else:
                preference_vendor = provider_field

        return {
            "preference": {
                "model": preference_model,
                "vendor": preference_vendor,
                "route": preference_route,
            },
            "fallbacks": self._mandate_fallbacks.copy(),
            "banned": self.mandate_config.get("defaults", {}).get("banned", []),
            "mandates": self.mandate_config.get("mandates", {})
        }


    def add_fallback_model(self, model_name: str, provider: Optional[str] = None):
        """Add a model to the fallback list."""
        fallback_entry = {"model": model_name, "provider": provider}

        if fallback_entry not in self._mandate_fallbacks:
            self._mandate_fallbacks.append(fallback_entry)
            logger.info(f"Added fallback model: {model_name}" +
                        (f" (provider: {provider})" if provider else ""))
        else:
            logger.info(f"Fallback model already exists: {model_name}")

    def clear_mandate(self):
        """Reset the mandate to defaults from the TOML file."""
        self.mandate_config = load_config("model_mandate.toml")
        self._mandate_preference = {"model": None, "provider": None}
        self._mandate_fallbacks = []
        logger.info("Model mandate cleared, reset to TOML defaults")

    def get_active_mandate_text(self) -> str:
        """Returns the raw text of the currently active model mandate."""
        try:
            with open("model_mandate.toml", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "No model_mandate.toml file found."
        except Exception as e:
            return f"Error reading model_mandate.toml: {e}"
