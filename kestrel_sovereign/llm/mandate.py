"""Model mandate management for LLM Service."""
import logging
from typing import Dict, Any, Optional, List

from kestrel_sovereign.config import load_config
from .provider_names import normalize_provider_name, provider_name_candidates

logger = logging.getLogger(__name__)


class ModelMandateMixin:
    """Mixin class providing model mandate methods for LLMService."""

    def _resolve_model_selector(
        self,
        selector: Optional[str],
        providers: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Optional[str]]:
        """Resolve a mandate selector to a provider/model pair.

        Supported selectors:
        - provider name: ``anthropic``
        - provider/model: ``anthropic/claude-sonnet-4-6``
        - exact model id: ``gpt-5-mini``
        - alias: ``cheap``
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
            provider_name, model_name = selector.split("/", 1)
            provider_name = normalize_provider_name(provider_name)
            return {
                "selector": selector,
                "provider": provider_name,
                "model": model_name,
            }

        for provider in providers:
            if provider.get("name") in provider_name_candidates(selector):
                model_name = provider.get("model")
                provider_name = provider.get("name")
                normalized = f"{provider_name}/{model_name}" if model_name else provider_name
                return {
                    "selector": normalized,
                    "provider": provider_name,
                    "model": model_name,
                }

        for provider in providers:
            if provider.get("model") == selector:
                provider_name = provider.get("name")
                normalized = f"{provider_name}/{selector}" if provider_name else selector
                return {
                    "selector": normalized,
                    "provider": provider_name,
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
        preference_provider = self._mandate_preference.get("provider")

        if preference_model is None:
            resolved_default = self._resolve_model_selector(self._get_default_mandate_selector())
            preference_model = resolved_default.get("model")
            preference_provider = resolved_default.get("provider")

        return {
            "preference": {
                "model": preference_model,
                "provider": preference_provider
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
