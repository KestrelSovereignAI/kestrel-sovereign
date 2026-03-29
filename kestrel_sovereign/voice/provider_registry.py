"""
Voice Provider Registry.

Manages TTS and STT provider registration, discovery, and routing.
Mirrors the pattern in kestrel_sovereign/llm/provider_registry.py.
"""
import importlib.util
import logging
from typing import Optional

from .base import TTSProvider, STTProvider

logger = logging.getLogger(__name__)


class VoiceProviderRegistry:
    """Registry for TTS and STT providers."""

    def __init__(self, config: dict):
        """Initialize the voice provider registry.

        Args:
            config: Voice configuration dictionary (the [voice] section from kestrel.toml).
        """
        self._tts_providers: dict[str, TTSProvider] = {}
        self._stt_providers: dict[str, STTProvider] = {}
        self._config = config
        self._initialized = False

    async def initialize(self) -> None:
        """Discover and initialize available providers based on config and installed packages.

        Iterates through the priority lists in config and attempts to register
        each provider. Providers that aren't available (missing packages, no API keys)
        are skipped with a warning.
        """
        if self._initialized:
            return

        tts_priority = self._config.get("tts_provider_priority", [])
        stt_priority = self._config.get("stt_provider_priority", [])

        for provider_name in tts_priority:
            try:
                provider = self._create_tts_provider(provider_name)
                if provider and await provider.is_available():
                    self.register_tts(provider)
                    logger.info(f"Initialized TTS provider: {provider_name}")
                else:
                    logger.warning(f"TTS provider '{provider_name}' not available. Skipping.")
            except Exception as e:
                logger.error(f"Failed to initialize TTS provider '{provider_name}': {e}")

        for provider_name in stt_priority:
            try:
                provider = self._create_stt_provider(provider_name)
                if provider and await provider.is_available():
                    self.register_stt(provider)
                    logger.info(f"Initialized STT provider: {provider_name}")
                else:
                    logger.warning(f"STT provider '{provider_name}' not available. Skipping.")
            except Exception as e:
                logger.error(f"Failed to initialize STT provider '{provider_name}': {e}")

        self._initialized = True

    def _create_tts_provider(self, name: str) -> Optional[TTSProvider]:
        """Create a TTS provider by name using importlib for optional deps.

        Concrete provider implementations will be added as they are developed
        (e.g., PiperTTSProvider, OpenAITTSProvider).

        Args:
            name: Provider name (e.g., "piper", "openai").

        Returns:
            TTSProvider instance or None if unknown.
        """
        provider_config = self._config.get(name, {})

        if name == "openai":
            from .openai_tts import OpenAITTSProvider
            return OpenAITTSProvider(config=provider_config)

        if name == "elevenlabs":
            if importlib.util.find_spec("elevenlabs") is not None:
                from .elevenlabs_tts import ElevenLabsTTSProvider
                provider_config = self._config.get("elevenlabs", {})
                return ElevenLabsTTSProvider(config=provider_config)
            else:
                logger.warning("ElevenLabs TTS requested but 'elevenlabs' package not installed.")
                return None

        if name == "piper":
            from .piper_tts import PiperTTSProvider
            piper_config = self._config.get("piper", {})
            return PiperTTSProvider(piper_config)

        logger.warning(f"No TTS provider implementation for '{name}' yet.")
        return None

    def _create_stt_provider(self, name: str) -> Optional[STTProvider]:
        """Create an STT provider by name using importlib for optional deps.

        Args:
            name: Provider name (e.g., "faster_whisper", "openai").

        Returns:
            STTProvider instance or None if unknown.
        """
        provider_config = self._config.get(name, {})

        if name == "openai":
            from .openai_stt import OpenAISTTProvider
            return OpenAISTTProvider(config=provider_config)

        logger.warning(f"No STT provider implementation for '{name}' yet.")
        return None

    def register_tts(self, provider: TTSProvider) -> None:
        """Register a TTS provider.

        Args:
            provider: TTSProvider instance to register.
        """
        self._tts_providers[provider.name] = provider

    def register_stt(self, provider: STTProvider) -> None:
        """Register an STT provider.

        Args:
            provider: STTProvider instance to register.
        """
        self._stt_providers[provider.name] = provider

    def get_tts(self, name: str) -> Optional[TTSProvider]:
        """Get a TTS provider by name.

        Args:
            name: Provider name.

        Returns:
            TTSProvider if found, None otherwise.
        """
        return self._tts_providers.get(name)

    def get_stt(self, name: str) -> Optional[STTProvider]:
        """Get an STT provider by name.

        Args:
            name: Provider name.

        Returns:
            STTProvider if found, None otherwise.
        """
        return self._stt_providers.get(name)

    def get_local_tts(self) -> list[TTSProvider]:
        """Return only local (privacy-safe) TTS providers."""
        return [p for p in self._tts_providers.values() if p.is_local]

    def get_local_stt(self) -> list[STTProvider]:
        """Return only local (privacy-safe) STT providers."""
        return [p for p in self._stt_providers.values() if p.is_local]

    def list_tts_providers(self) -> list[str]:
        """List registered TTS provider names."""
        return list(self._tts_providers.keys())

    def list_stt_providers(self) -> list[str]:
        """List registered STT provider names."""
        return list(self._stt_providers.keys())
