"""
Voice Feature — TTS synthesis, STT transcription, voice selection.

Exposes list_voices, set_voice, speak, and transcribe tools to the agent
orchestrator. Privacy-gated: cloud providers are blocked in ephemeral/isolated
modes. Audio storage respects privacy config (none/temp/scrubbed/full).
Audio bytes are stored via the storage layer and referenced by
content_hash — raw bytes are never returned in tool results.
"""

import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.privacy import PrivacyConfig, PRIVACY_PRESETS
from kestrel_sovereign.tools.base import ToolCategory
from kestrel_sovereign.voice.base import VoiceConfig, VoiceInfo, TTSProvider, STTProvider
from kestrel_sovereign.voice.provider_registry import VoiceProviderRegistry

logger = logging.getLogger(__name__)

# Cloud provider names used in error messages
_CLOUD_TTS_NAMES = {"openai", "elevenlabs"}
_CLOUD_STT_NAMES = {"openai", "deepgram"}


class VoicePrivacyError(Exception):
    """Raised when a voice operation is blocked by the current privacy mode."""


class VoiceFeature(Feature):
    """Voice capabilities — TTS synthesis, STT transcription, voice selection."""

    @property
    def tool_description(self) -> str:
        return (
            "Voice capabilities — text-to-speech synthesis, speech-to-text transcription, "
            "voice selection and management for agent identity"
        )

    async def initialize(self):
        """Lazy-load VoiceProviderRegistry from agent config."""
        self._voice_registry = None
        self._voice_config = VoiceConfig()

        # Load voice config from agent identity if available
        identity = getattr(self.agent, "identity", None)
        if identity and hasattr(identity, "voice_config"):
            stored = identity.voice_config
            if isinstance(stored, dict):
                self._voice_config = VoiceConfig.from_dict(stored)
            elif isinstance(stored, VoiceConfig):
                self._voice_config = stored

    async def _ensure_registry(self) -> VoiceProviderRegistry:
        """Lazily initialize and return the voice provider registry."""
        if self._voice_registry is not None:
            return self._voice_registry

        # Pull voice config section from agent config
        agent_config = getattr(self.agent, "config", None)
        voice_section: dict = {}
        if agent_config:
            if hasattr(agent_config, "get"):
                voice_section = agent_config.get("voice", {})
            elif hasattr(agent_config, "voice"):
                voice_section = agent_config.voice or {}

        self._voice_registry = VoiceProviderRegistry(voice_section)
        await self._voice_registry.initialize()
        return self._voice_registry

    # ------------------------------------------------------------------
    # Privacy gates
    # ------------------------------------------------------------------

    def _get_privacy_config(self) -> Optional[PrivacyConfig]:
        """Get the current privacy config from the agent's privacy agent."""
        privacy_agent = getattr(self.agent, "privacy_agent", None)
        if privacy_agent is None:
            return None
        return getattr(privacy_agent, "privacy_config", None)

    def _get_privacy_mode_name(self) -> str:
        """Get the current privacy preset name (e.g. 'ephemeral', 'normal')."""
        config = self._get_privacy_config()
        if config is None:
            return "normal"
        for name, preset in PRIVACY_PRESETS.items():
            if (config.storage == preset.storage
                    and config.llm_location == preset.llm_location
                    and config.shareable == preset.shareable):
                return name
        return "custom"

    def _cloud_allowed(self) -> bool:
        """Check whether cloud providers are allowed under current privacy mode."""
        config = self._get_privacy_config()
        if config is None:
            return True
        return config.allows_cloud_llm()

    def _get_audio_storage_policy(self) -> str:
        """Determine audio storage policy based on privacy config.

        Returns one of: "none", "temp", "scrubbed", "full".
        - none: Audio bytes NEVER written to disk (ephemeral).
        - temp: Audio in temp buffer, deleted on session end (isolated).
        - scrubbed: Audio not permanently stored, metadata scrubbed (anonymous).
        - full: Audio may be cached/stored normally (normal/public).
        """
        config = self._get_privacy_config()
        if config is None:
            return "full"
        return config.storage

    async def _get_tts_provider(self) -> TTSProvider:
        """Get TTS provider respecting privacy mode."""
        registry = await self._ensure_registry()
        if not self._cloud_allowed():
            mode_name = self._get_privacy_mode_name()
            # If the configured provider is a cloud provider, block it with a clear message
            configured = self._voice_config.tts_provider
            if configured:
                tts = registry.get_tts(configured)
                if tts and not tts.is_local:
                    raise VoicePrivacyError(
                        f"Cannot use {configured.title()} TTS in {mode_name} privacy mode. "
                        f"Install piper-tts for local TTS, or switch to 'anonymous' or higher privacy mode."
                    )
            providers = registry.get_local_tts()
            if not providers:
                raise VoicePrivacyError(
                    f"No local TTS provider available in {mode_name} privacy mode. "
                    f"Install piper-tts for local TTS, or switch to 'anonymous' or higher privacy mode."
                )
            return providers[0]
        # Use configured provider, fall back to first available
        if self._voice_config.tts_provider:
            provider = registry.get_tts(self._voice_config.tts_provider)
            if provider:
                return provider
        # Fall back to first registered TTS provider
        for name in registry.list_tts_providers():
            provider = registry.get_tts(name)
            if provider:
                return provider
        raise VoicePrivacyError("No TTS provider available. Configure a voice provider.")

    async def _get_stt_provider(self) -> STTProvider:
        """Get STT provider respecting privacy mode."""
        registry = await self._ensure_registry()
        if not self._cloud_allowed():
            mode_name = self._get_privacy_mode_name()
            # If the configured provider is a cloud provider, block it with a clear message
            configured = self._voice_config.stt_provider
            if configured:
                stt = registry.get_stt(configured)
                if stt and not stt.is_local:
                    raise VoicePrivacyError(
                        f"Cannot use {configured.title()} STT in {mode_name} privacy mode. "
                        f"Install faster-whisper for local STT, or switch to 'anonymous' or higher privacy mode."
                    )
            providers = registry.get_local_stt()
            if not providers:
                raise VoicePrivacyError(
                    f"No local STT provider available in {mode_name} privacy mode. "
                    f"Install faster-whisper for local STT, or switch to 'anonymous' or higher privacy mode."
                )
            return providers[0]
        # Use configured provider, fall back to first available
        if self._voice_config.stt_provider:
            provider = registry.get_stt(self._voice_config.stt_provider)
            if provider:
                return provider
        for name in registry.list_stt_providers():
            provider = registry.get_stt(name)
            if provider:
                return provider
        raise VoicePrivacyError("No STT provider available. Configure a voice provider.")

    def is_provider_allowed(self, provider_name: str, provider_type: str = "tts") -> bool:
        """Check if a specific provider is allowed under current privacy mode.

        Args:
            provider_name: Name of the provider (e.g., "openai", "piper").
            provider_type: "tts" or "stt".

        Returns:
            True if the provider is allowed.
        """
        if self._cloud_allowed():
            return True
        registry = getattr(self, "_voice_registry", None)
        if registry is None:
            return True
        if provider_type == "tts":
            provider = registry.get_tts(provider_name)
        else:
            provider = registry.get_stt(provider_name)
        if provider is None:
            return True
        return provider.is_local

    async def on_privacy_mode_changed(self) -> Optional[dict]:
        """React to a privacy mode change. Auto-switch voice providers if needed.

        Called by the privacy mode change endpoint. If switching to a mode that
        blocks cloud and the current voice is a cloud provider, auto-fall back to
        a local provider. If no local provider available, clear the voice config.

        Returns:
            Dict with voice_switched info, or None if no change needed.
        """
        if self._cloud_allowed():
            return None

        registry = await self._ensure_registry()
        result = {}

        # Check TTS provider
        if self._voice_config.tts_provider:
            tts = registry.get_tts(self._voice_config.tts_provider)
            if tts and not tts.is_local:
                local_tts = registry.get_local_tts()
                if local_tts:
                    old_provider = self._voice_config.tts_provider
                    self._voice_config.tts_provider = local_tts[0].name
                    self._voice_config.tts_voice_id = ""
                    result["tts_switched"] = {
                        "from": old_provider,
                        "to": local_tts[0].name,
                    }
                    logger.info("Auto-switched TTS from %s to %s due to privacy mode", old_provider, local_tts[0].name)
                else:
                    old_provider = self._voice_config.tts_provider
                    self._voice_config.tts_provider = ""
                    self._voice_config.tts_voice_id = ""
                    result["tts_switched"] = {"from": old_provider, "to": None}
                    logger.warning("No local TTS provider available; TTS disabled due to privacy mode")

        # Check STT provider
        if self._voice_config.stt_provider:
            stt = registry.get_stt(self._voice_config.stt_provider)
            if stt and not stt.is_local:
                local_stt = registry.get_local_stt()
                if local_stt:
                    old_provider = self._voice_config.stt_provider
                    self._voice_config.stt_provider = local_stt[0].name
                    result["stt_switched"] = {
                        "from": old_provider,
                        "to": local_stt[0].name,
                    }
                    logger.info("Auto-switched STT from %s to %s due to privacy mode", old_provider, local_stt[0].name)
                else:
                    old_provider = self._voice_config.stt_provider
                    self._voice_config.stt_provider = ""
                    result["stt_switched"] = {"from": old_provider, "to": None}
                    logger.warning("No local STT provider available; STT disabled due to privacy mode")

        # Persist updated config to agent identity
        if result:
            identity = getattr(self.agent, "identity", None)
            if identity and hasattr(identity, "voice_config"):
                identity.voice_config = self._voice_config.to_dict()

        return result if result else None

    @staticmethod
    def biometric_warning() -> str:
        """Warning message about voice data being biometric."""
        return (
            "WARNING: Voice data is biometric information. Enabling cloud voice providers "
            "will send voice data to third-party services. This data may be used for "
            "processing and cannot be fully recalled once transmitted."
        )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool("list_voices", "List available TTS voices, optionally filtered by provider", ToolCategory.SYSTEM)
    async def list_voices(self, provider: str = "") -> dict:
        """List available voices.

        Args:
            provider: Filter to a specific provider (e.g., "openai", "piper"). Empty = all.
        """
        registry = await self._ensure_registry()
        cloud_ok = self._cloud_allowed()
        voices: list[dict] = []

        tts_names = registry.list_tts_providers()
        if provider:
            tts_names = [n for n in tts_names if n == provider]

        for name in tts_names:
            tts = registry.get_tts(name)
            if tts is None:
                continue
            # Skip cloud providers when privacy blocks them
            if not cloud_ok and not tts.is_local:
                continue
            try:
                provider_voices = await tts.list_voices()
                for v in provider_voices:
                    voices.append(asdict(v) if isinstance(v, VoiceInfo) else v)
            except Exception as exc:
                logger.warning("Failed to list voices from %s: %s", name, exc)

        return {"voices": voices, "count": len(voices)}

    @tool("set_voice", "Set the agent's TTS voice for spoken responses", ToolCategory.SYSTEM)
    async def set_voice(self, voice_id: str, provider: str = "") -> dict:
        """Set the agent's voice.

        Args:
            voice_id: The voice identifier (e.g., "nova", "en_US-lessac-medium")
            provider: Provider name. If empty, auto-detect from voice_id.
        """
        registry = await self._ensure_registry()

        # Auto-detect provider if not specified
        resolved_provider = provider
        if not resolved_provider:
            for name in registry.list_tts_providers():
                tts = registry.get_tts(name)
                if tts is None:
                    continue
                try:
                    provider_voices = await tts.list_voices()
                    if any(v.voice_id == voice_id for v in provider_voices):
                        resolved_provider = name
                        break
                except Exception:
                    continue

        if not resolved_provider:
            return {"success": False, "error": f"Could not find voice '{voice_id}' in any provider."}

        # Validate provider is accessible under current privacy mode
        tts = registry.get_tts(resolved_provider)
        if tts and not self._cloud_allowed() and not tts.is_local:
            return {
                "success": False,
                "error": f"Provider '{resolved_provider}' is a cloud service blocked by current privacy mode.",
            }

        self._voice_config.tts_provider = resolved_provider
        self._voice_config.tts_voice_id = voice_id

        # Persist to agent identity if available
        identity = getattr(self.agent, "identity", None)
        if identity and hasattr(identity, "voice_config"):
            identity.voice_config = self._voice_config.to_dict()

        return {
            "success": True,
            "voice_id": voice_id,
            "provider": resolved_provider,
        }

    @tool("speak", "Synthesize speech from text using the agent's configured voice", ToolCategory.SYSTEM)
    async def speak(self, text: str) -> dict:
        """Synthesize speech from text.

        Args:
            text: The text to speak aloud
        """
        tts = await self._get_tts_provider()
        voice_id = self._voice_config.tts_voice_id
        if not voice_id:
            # Use first available voice
            voices = await tts.list_voices()
            if not voices:
                return {"success": False, "error": "No voices available on the current TTS provider."}
            voice_id = voices[0].voice_id

        audio_bytes = await tts.synthesize(
            text=text,
            voice_id=voice_id,
            model=self._voice_config.tts_model,
            output_format=self._voice_config.output_format,
        )

        # Store audio respecting privacy policy
        storage_policy = self._get_audio_storage_policy()
        content_hash = ""

        if storage_policy == "none":
            # Ephemeral: audio bytes NEVER written to disk, served from memory only
            logger.debug("Ephemeral mode: audio bytes not persisted.")
        elif storage_policy == "temp":
            # Isolated: audio in temp buffer, deleted on session end
            storage = getattr(self.agent, "storage", None)
            if storage and hasattr(storage, "store_file"):
                ext = self._voice_config.output_format or "opus"
                content_hash = await storage.store_file(
                    audio_bytes,
                    f"speech.{ext}",
                    metadata={
                        "type": "tts_audio",
                        "format": ext,
                        "voice_id": voice_id,
                        "provider": tts.name,
                        "text_length": len(text),
                        "storage_policy": "temp",
                    },
                )
        elif storage_policy == "scrubbed":
            # Anonymous: audio not permanently stored, metadata scrubbed
            storage = getattr(self.agent, "storage", None)
            if storage and hasattr(storage, "store_file"):
                ext = self._voice_config.output_format or "opus"
                content_hash = await storage.store_file(
                    audio_bytes,
                    f"speech.{ext}",
                    metadata={
                        "type": "tts_audio",
                        "format": ext,
                        "text_length": len(text),
                        "storage_policy": "scrubbed",
                        # voice_id and provider intentionally omitted to avoid speaker identification
                    },
                )
        else:
            # Full storage (normal/public)
            storage = getattr(self.agent, "storage", None)
            if storage and hasattr(storage, "store_file"):
                ext = self._voice_config.output_format or "opus"
                content_hash = await storage.store_file(
                    audio_bytes,
                    f"speech.{ext}",
                    metadata={
                        "type": "tts_audio",
                        "format": ext,
                        "voice_id": voice_id,
                        "provider": tts.name,
                        "text_length": len(text),
                    },
                )
            else:
                logger.warning("No storage available; audio bytes not persisted.")

        return {
            "success": True,
            "content_hash": content_hash,
            "format": self._voice_config.output_format,
            "voice_id": voice_id,
            "provider": tts.name,
            "text_length": len(text),
            "audio_size": len(audio_bytes),
            "storage_policy": storage_policy,
        }

    @tool("transcribe", "Transcribe audio to text", ToolCategory.SYSTEM)
    async def transcribe(self, audio_content_hash: str) -> dict:
        """Transcribe audio file to text.

        Args:
            audio_content_hash: Content hash of audio file (retrievable via /api/files/{hash})
        """
        # Retrieve audio from storage
        storage = getattr(self.agent, "storage", None)
        if not storage or not hasattr(storage, "retrieve_file"):
            return {"success": False, "error": "Storage not available for audio retrieval."}

        audio_bytes = await storage.retrieve_file(audio_content_hash)
        if audio_bytes is None:
            return {"success": False, "error": f"Audio file not found: {audio_content_hash}"}

        # Determine audio format from metadata
        audio_format = "opus"
        if hasattr(storage, "get_file_metadata"):
            meta = await storage.get_file_metadata(audio_content_hash)
            if meta and isinstance(meta, dict):
                audio_format = meta.get("format", audio_format)

        stt = await self._get_stt_provider()
        transcript = await stt.transcribe(
            audio=audio_bytes,
            language="",
            audio_format=audio_format,
        )

        return {
            "success": True,
            "text": transcript,
            "provider": stt.name,
            "audio_content_hash": audio_content_hash,
        }
