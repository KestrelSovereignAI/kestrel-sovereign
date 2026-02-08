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
        # Use current provider if not specified
        if provider is None:
            if self.providers and len(self.providers) > 0:
                provider = self.providers[0].get("name", "openai")
            else:
                provider = "openai"

        # Use current model if not specified
        if model is None:
            if self.providers and len(self.providers) > 0:
                model = self.providers[0].get("model", "gpt-4")
            else:
                model = "gpt-4"

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
        # Use current provider if not specified
        if provider is None:
            if self.providers and len(self.providers) > 0:
                provider = self.providers[0].get("name", "openai")
            else:
                provider = "openai"

        # Use current model if not specified
        if model is None:
            if self.providers and len(self.providers) > 0:
                model = self.providers[0].get("model", "gpt-4")
            else:
                model = "gpt-4"

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

    def get_current_provider_and_model(self) -> Tuple[str, str]:
        """
        Get the current provider and model being used.

        Returns:
            Tuple of (provider_name, model_id)
        """
        if self.providers and len(self.providers) > 0:
            provider = self.providers[0].get("name", "openai")
            model = self.providers[0].get("model", "gpt-4")
        else:
            provider = "openai"
            model = "gpt-4"

        return provider, model
