"""Model mandate management for LLM Service."""
import logging
from typing import Dict, Any, Optional

from kestrel_sovereign.config import load_config

logger = logging.getLogger(__name__)


class ModelMandateMixin:
    """Mixin class providing model mandate methods for LLMService."""

    def get_current_mandate(self) -> Dict[str, Any]:
        """Get the current model mandate configuration."""
        preference_model = self._mandate_preference.get("model")
        preference_provider = self._mandate_preference.get("provider")

        if preference_model is None:
            preference_model = self.mandate_config.get("defaults", {}).get("preferred")
            if preference_model:
                for p in self.providers:
                    if p.get("name") == preference_model or p.get("model") == preference_model:
                        preference_provider = p.get("name")
                        break

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
            with open("model_mandate.toml", "r") as f:
                return f.read()
        except FileNotFoundError:
            return "No model_mandate.toml file found."
        except Exception as e:
            return f"Error reading model_mandate.toml: {e}"
