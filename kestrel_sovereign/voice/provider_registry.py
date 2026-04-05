"""
Voice Provider Registry.

Manages TTS and STT provider registration, discovery, and routing.
Mirrors the pattern in kestrel_sovereign/llm/provider_registry.py.

External packages can register voice providers via entry_points::

    [project.entry-points."kestrel_sovereign.voice_providers"]
    ElevenLabsTTS = "kestrel_voice_elevenlabs:ElevenLabsTTSProvider"
    ElevenLabsSTT = "kestrel_voice_elevenlabs:ElevenLabsSTTProvider"
"""
import logging
from typing import Optional

from kestrel_sovereign.entrypoints import discover_entry_point_classes
from .base import TTSProvider, STTProvider

logger = logging.getLogger(__name__)

VOICE_PROVIDER_ENTRY_POINT_GROUP = "kestrel_sovereign.voice_providers"


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

        After built-in providers, scans entry_points for external voice providers.
        Built-in providers win on name collisions.
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

        # Phase 2: Discover external providers via entry_points
        await self._discover_entrypoint_providers()

        self._initialized = True

    async def _discover_entrypoint_providers(self) -> None:
        """Scan entry_points for external TTS/STT providers.

        Discovered providers that are subclasses of TTSProvider or STTProvider
        are instantiated with config and registered. Built-in providers
        (already registered) win on name collisions.
        """
        # Single scan — entry points can provide either TTS or STT (or both)
        # We check against both base classes
        tts_classes = discover_entry_point_classes(VOICE_PROVIDER_ENTRY_POINT_GROUP, TTSProvider)
        stt_classes = discover_entry_point_classes(VOICE_PROVIDER_ENTRY_POINT_GROUP, STTProvider)

        for ep_name, cls in tts_classes.items():
            try:
                provider_config = self._config.get(ep_name, {})
                provider = cls(config=provider_config)
                if provider.name in self._tts_providers:
                    logger.debug(f"Skipping entry_point TTS '{ep_name}': built-in '{provider.name}' already registered")
                    continue
                if await provider.is_available():
                    self.register_tts(provider)
                    logger.info(f"Registered entry_point TTS provider: {ep_name}")
                else:
                    logger.debug(f"Entry_point TTS provider '{ep_name}' not available, skipping")
            except Exception as e:
                logger.warning(f"Failed to load entry_point TTS provider '{ep_name}': {e}")

        for ep_name, cls in stt_classes.items():
            try:
                provider_config = self._config.get(ep_name, {})
                provider = cls(config=provider_config)
                if provider.name in self._stt_providers:
                    logger.debug(f"Skipping entry_point STT '{ep_name}': built-in '{provider.name}' already registered")
                    continue
                if await provider.is_available():
                    self.register_stt(provider)
                    logger.info(f"Registered entry_point STT provider: {ep_name}")
                else:
                    logger.debug(f"Entry_point STT provider '{ep_name}' not available, skipping")
            except Exception as e:
                logger.warning(f"Failed to load entry_point STT provider '{ep_name}': {e}")

    # Cloud voice providers (openai, elevenlabs, deepgram) are discovered
    # via entry_points from their respective packages:
    #   kestrel-voice-openai, kestrel-voice-elevenlabs, kestrel-voice-deepgram

    def _create_tts_provider(self, name: str) -> Optional[TTSProvider]:
        """Create a TTS provider by name.

        Local providers (piper) are imported from core. Cloud providers
        (openai, elevenlabs) are discovered via entry_points.
        """
        if name == "piper":
            try:
                from .piper_tts import PiperTTSProvider
                piper_config = self._config.get("piper", {})
                return PiperTTSProvider(piper_config)
            except ImportError:
                logger.warning("piper-tts package not installed. Skipping piper TTS.")
                return None

        # Cloud providers are discovered via entry_points in _discover_entrypoint_providers()
        logger.debug(f"TTS provider '{name}' not a built-in; will check entry_points.")
        return None

    def _create_stt_provider(self, name: str) -> Optional[STTProvider]:
        """Create an STT provider by name.

        Local providers (faster_whisper) are imported from core. Cloud
        providers (openai, deepgram) are discovered via entry_points.
        """
        if name == "faster_whisper":
            try:
                from .faster_whisper_stt import FasterWhisperSTTProvider
                provider_config = self._config.get("faster_whisper", {})
                return FasterWhisperSTTProvider(config=provider_config)
            except ImportError:
                logger.warning("faster-whisper package not installed. Skipping faster_whisper STT.")
                return None

        # Cloud providers are discovered via entry_points in _discover_entrypoint_providers()
        logger.debug(f"STT provider '{name}' not a built-in; will check entry_points.")
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
