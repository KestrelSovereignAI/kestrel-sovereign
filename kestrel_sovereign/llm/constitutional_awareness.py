"""Constitutional awareness for LLM Service."""
import logging
from typing import Optional, Tuple

from .constitutional_profile import (
    get_profile_service,
    ConstitutionalProfile,
    StateOfMind,
    PromptAdaptation
)

logger = logging.getLogger(__name__)


class ConstitutionalAwarenessMixin:
    """Mixin class providing constitutional awareness methods for LLMService."""

    def _init_constitutional_profiles(self) -> None:
        """Initialize constitutional profile service."""
        # Get the global singleton - it will lazy-load on first access
        self._profile_service = get_profile_service()
        logger.debug("Constitutional profile service initialized")

    def get_current_provider_and_model(self) -> Tuple[str, str]:
        """
        Get the current provider and model being used.

        Uses get_model_preference() (mandate system) as single source of truth,
        falling back to first configured provider.

        Returns:
            Tuple of (provider_name, model_id)
        """
        pref = self.get_model_preference()
        provider = pref.get("provider")
        model = pref.get("model")

        if not provider or not model:
            if self.providers and len(self.providers) > 0:
                provider = provider or self.providers[0].get("name", "openai")
                model = model or self.providers[0].get("model", "auto")
            else:
                provider = provider or "openai"
                model = model or "auto"

        return provider, model

    def get_constitutional_profile(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> ConstitutionalProfile:
        """
        Get constitutional profile for current or specified provider/model.

        Args:
            provider: Provider name (defaults to current provider)
            model: Model ID (defaults to current model)

        Returns:
            ConstitutionalProfile for the provider/model
        """
        if provider is None or model is None:
            current_provider, current_model = self.get_current_provider_and_model()
            provider = provider or current_provider
            model = model or current_model

        return self._profile_service.get_profile_for_model(model, provider)

    def get_state_of_mind(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> StateOfMind:
        """
        Get current state of mind based on active provider/model.

        Args:
            provider: Provider name (defaults to current provider)
            model: Model ID (defaults to current model)

        Returns:
            StateOfMind descriptor
        """
        if provider is None or model is None:
            current_provider, current_model = self.get_current_provider_and_model()
            provider = provider or current_provider
            model = model or current_model

        return self._profile_service.get_state_of_mind(provider, model)

    def get_prompt_adaptation(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> PromptAdaptation:
        """
        Get prompt adaptation strategy for current or specified provider/model.

        Args:
            provider: Provider name (defaults to current provider)
            model: Model ID (defaults to current model)

        Returns:
            PromptAdaptation strategy
        """
        profile = self.get_constitutional_profile(provider, model)
        return profile.prompt_adaptation
