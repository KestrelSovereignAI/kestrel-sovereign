"""
Base classes for voice providers (TTS and STT).

Defines the abstract contracts that all voice providers must implement.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class VoiceInfo:
    """Metadata about an available voice."""
    voice_id: str           # Provider-specific ID (e.g., "nova", "en_US-lessac-medium")
    name: str               # Human-readable name
    provider: str           # Provider name (e.g., "openai", "piper")
    language: str = "en"    # ISO 639-1 language code
    gender: str = "neutral" # "masculine", "feminine", "neutral"
    preview_url: str = ""   # URL to sample audio (optional)


@dataclass
class VoiceConfig:
    """Voice configuration for an agent. Persisted in AgentIdentityPackage."""
    tts_provider: str = ""
    tts_voice_id: str = ""
    tts_model: str = ""
    stt_provider: str = ""
    stt_model: str = ""
    sample_rate: int = 24000
    output_format: str = "opus"  # opus, mp3, pcm, wav

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceConfig":
        """Deserialize from dictionary, ignoring unknown keys."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


class TTSProvider(ABC):
    """Abstract base for text-to-speech providers."""

    name: str
    is_local: bool  # True = privacy-safe, no cloud calls

    @abstractmethod
    async def synthesize(self, text: str, voice_id: str, model: str = "",
                         output_format: str = "opus") -> bytes:
        """Synthesize text to audio bytes.

        Args:
            text: Text to synthesize.
            voice_id: Provider-specific voice identifier.
            model: Optional model override.
            output_format: Audio format (opus, mp3, pcm, wav).

        Returns:
            Audio data as bytes.
        """

    @abstractmethod
    async def synthesize_stream(self, text: str, voice_id: str, model: str = "",
                                output_format: str = "opus") -> AsyncIterator[bytes]:
        """Stream synthesized audio chunks.

        Args:
            text: Text to synthesize.
            voice_id: Provider-specific voice identifier.
            model: Optional model override.
            output_format: Audio format (opus, mp3, pcm, wav).

        Yields:
            Audio data chunks as bytes.
        """

    @abstractmethod
    async def list_voices(self) -> list[VoiceInfo]:
        """List available voices for this provider.

        Returns:
            List of VoiceInfo for each available voice.
        """

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this provider is available.

        Checks: (a) required package installed, (b) required API key present
        or local model available.

        Returns:
            True if the provider can be used.
        """


class STTProvider(ABC):
    """Abstract base for speech-to-text providers."""

    name: str
    is_local: bool  # True = privacy-safe, no cloud calls

    @abstractmethod
    async def transcribe(self, audio: bytes, language: str = "",
                         audio_format: str = "opus") -> str:
        """Transcribe audio bytes to text.

        Args:
            audio: Audio data as bytes.
            language: Optional ISO 639-1 language hint.
            audio_format: Audio format (opus, mp3, pcm, wav).

        Returns:
            Transcribed text.
        """

    @abstractmethod
    async def transcribe_stream(self, audio_stream: AsyncIterator[bytes],
                                language: str = "") -> AsyncIterator[str]:
        """Stream transcription from audio chunks.

        Args:
            audio_stream: Async iterator of audio chunks.
            language: Optional ISO 639-1 language hint.

        Yields:
            Transcribed text segments.
        """

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this provider is available.

        Checks: (a) required package installed, (b) required API key present
        or local model available.

        Returns:
            True if the provider can be used.
        """
