"""
ElevenLabs TTS Provider.

Cloud-based text-to-speech using the ElevenLabs API.
Supports voice cloning and a large library of custom voices.
"""
import importlib.util
import logging
import os
from typing import AsyncIterator

from .base import TTSProvider, VoiceInfo

logger = logging.getLogger(__name__)

# Map canonical output formats to ElevenLabs output format strings.
_FORMAT_MAP = {
    "mp3": "mp3_44100_128",
    "pcm": "pcm_24000",
    "wav": "pcm_24000",  # ElevenLabs returns raw PCM; caller wraps if needed
    "opus": "mp3_44100_128",  # ElevenLabs has no native opus; fall back to mp3
    "ulaw": "ulaw_8000",
}

# ElevenLabs-specific sample rate mapping for PCM formats.
_PCM_SAMPLE_RATES = {
    16000: "pcm_16000",
    22050: "pcm_22050",
    24000: "pcm_24000",
    44100: "pcm_44100",
}


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs cloud TTS provider."""

    name = "elevenlabs"
    is_local = False  # Cloud provider

    MODELS = [
        "eleven_multilingual_v2",
        "eleven_turbo_v2_5",
        "eleven_monolingual_v1",
    ]

    def __init__(self, config: dict | None = None):
        """Initialize ElevenLabs TTS provider.

        Args:
            config: The [voice.elevenlabs] section from kestrel.toml.
        """
        self._config = config or {}
        self._model = self._config.get("model", self.MODELS[0])
        self._default_voice_id = self._config.get("default_voice_id", "")
        self._client = None

    def _get_client(self):
        """Lazily create the async ElevenLabs client."""
        if self._client is None:
            from elevenlabs.client import AsyncElevenLabs

            api_key = os.environ.get("ELEVENLABS_API_KEY", "")
            self._client = AsyncElevenLabs(api_key=api_key)
        return self._client

    @staticmethod
    def _resolve_format(output_format: str) -> str:
        """Map a canonical format name to an ElevenLabs output_format string."""
        return _FORMAT_MAP.get(output_format, "mp3_44100_128")

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        model: str = "",
        output_format: str = "opus",
    ) -> bytes:
        """Synthesize text to audio bytes via ElevenLabs API."""
        voice_id = voice_id or self._default_voice_id
        if not voice_id:
            raise ValueError(
                "voice_id is required. Set default_voice_id in [voice.elevenlabs] "
                "config or pass voice_id explicitly."
            )

        model_id = model or self._model
        el_format = self._resolve_format(output_format)
        client = self._get_client()

        logger.info(
            "ElevenLabs synthesize: %d chars, voice=%s, model=%s, format=%s",
            len(text),
            voice_id,
            model_id,
            el_format,
        )

        audio = await client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
            output_format=el_format,
        )

        # The SDK returns an async iterator of bytes; collect them.
        if isinstance(audio, bytes):
            return audio

        chunks = []
        async for chunk in audio:
            chunks.append(chunk)
        return b"".join(chunks)

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str,
        model: str = "",
        output_format: str = "opus",
    ) -> AsyncIterator[bytes]:
        """Stream synthesized audio chunks from ElevenLabs."""
        voice_id = voice_id or self._default_voice_id
        if not voice_id:
            raise ValueError(
                "voice_id is required. Set default_voice_id in [voice.elevenlabs] "
                "config or pass voice_id explicitly."
            )

        model_id = model or self._model
        el_format = self._resolve_format(output_format)
        client = self._get_client()

        logger.info(
            "ElevenLabs synthesize_stream: %d chars, voice=%s, model=%s, format=%s",
            len(text),
            voice_id,
            model_id,
            el_format,
        )

        audio_stream = await client.text_to_speech.convert_as_stream(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
            output_format=el_format,
        )

        async for chunk in audio_stream:
            yield chunk

    async def list_voices(self) -> list[VoiceInfo]:
        """List available voices including user's cloned voices."""
        client = self._get_client()
        response = await client.voices.get_all()

        voices = []
        for voice in response.voices:
            labels = getattr(voice, "labels", {}) or {}
            gender = labels.get("gender", "neutral") if isinstance(labels, dict) else "neutral"
            language = labels.get("language", "en") if isinstance(labels, dict) else "en"
            preview = getattr(voice, "preview_url", "") or ""

            voices.append(
                VoiceInfo(
                    voice_id=voice.voice_id,
                    name=voice.name,
                    provider=self.name,
                    language=language,
                    gender=gender,
                    preview_url=preview,
                )
            )

        return voices

    async def is_available(self) -> bool:
        """Check if elevenlabs package is installed and API key is set."""
        if importlib.util.find_spec("elevenlabs") is None:
            return False
        if not os.environ.get("ELEVENLABS_API_KEY"):
            return False
        return True
