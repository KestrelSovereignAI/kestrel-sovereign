"""
OpenAI TTS Provider.

Supports tts-1 (fast) and tts-1-hd (high quality) with 6 built-in voices.
Uses the openai AsyncOpenAI client for async synthesis.
"""
import logging
import os
import time
from typing import AsyncIterator

import openai

from .base import TTSProvider, VoiceInfo

logger = logging.getLogger(__name__)


class OpenAITTSProvider(TTSProvider):
    """OpenAI text-to-speech provider using tts-1 and tts-1-hd models."""

    name = "openai"
    is_local = False

    VOICES = {
        "alloy": VoiceInfo(voice_id="alloy", name="Alloy", provider="openai", gender="neutral",
                           age="young", energy="warm", accent="american"),
        "echo": VoiceInfo(voice_id="echo", name="Echo", provider="openai", gender="masculine",
                          age="middle", energy="calm", accent="american"),
        "fable": VoiceInfo(voice_id="fable", name="Fable", provider="openai", gender="neutral",
                           age="middle", energy="warm", accent="british"),
        "nova": VoiceInfo(voice_id="nova", name="Nova", provider="openai", gender="feminine",
                          age="young", energy="warm", accent="american"),
        "onyx": VoiceInfo(voice_id="onyx", name="Onyx", provider="openai", gender="masculine",
                          age="mature", energy="authoritative", accent="american"),
        "shimmer": VoiceInfo(voice_id="shimmer", name="Shimmer", provider="openai", gender="feminine",
                             age="middle", energy="energetic", accent="american"),
    }

    MODELS = ["tts-1", "tts-1-hd"]

    FORMAT_MAP = {"opus": "opus", "mp3": "mp3", "wav": "wav", "pcm": "pcm"}

    def __init__(self, config: dict | None = None):
        """Initialize the OpenAI TTS provider.

        Args:
            config: Provider config from [voice.openai] section of kestrel.toml.
        """
        config = config or {}
        self._tts_model = config.get("tts_model", "tts-1-hd")
        self._default_voice = config.get("default_voice", "nova")
        api_key = os.environ.get("OPENAI_API_KEY")
        self._client = openai.AsyncOpenAI(api_key=api_key) if api_key else None

    async def synthesize(self, text: str, voice_id: str, model: str = "",
                         output_format: str = "opus") -> bytes:
        """Synthesize text to audio bytes via OpenAI TTS API."""
        if not self._client:
            raise RuntimeError("OpenAI API key not configured")

        effective_model = model or self._tts_model
        effective_voice = voice_id or self._default_voice
        effective_format = self.FORMAT_MAP.get(output_format, "opus")

        start = time.monotonic()
        response = await self._client.audio.speech.create(
            model=effective_model,
            voice=effective_voice,
            input=text,
            response_format=effective_format,
        )
        audio_bytes = response.content
        duration = time.monotonic() - start

        logger.info(
            "OpenAI TTS synthesized: model=%s voice=%s format=%s text_len=%d audio_bytes=%d duration=%.2fs",
            effective_model, effective_voice, effective_format, len(text), len(audio_bytes), duration,
        )
        return audio_bytes

    async def synthesize_stream(self, text: str, voice_id: str, model: str = "",
                                output_format: str = "opus") -> AsyncIterator[bytes]:
        """Stream synthesized audio chunks from OpenAI TTS API."""
        if not self._client:
            raise RuntimeError("OpenAI API key not configured")

        effective_model = model or self._tts_model
        effective_voice = voice_id or self._default_voice
        effective_format = self.FORMAT_MAP.get(output_format, "opus")

        start = time.monotonic()
        total_bytes = 0

        async with self._client.audio.speech.with_streaming_response.create(
            model=effective_model,
            voice=effective_voice,
            input=text,
            response_format=effective_format,
        ) as response:
            async for chunk in response.iter_bytes(chunk_size=4096):
                total_bytes += len(chunk)
                yield chunk

        duration = time.monotonic() - start
        logger.info(
            "OpenAI TTS streamed: model=%s voice=%s format=%s text_len=%d audio_bytes=%d duration=%.2fs",
            effective_model, effective_voice, effective_format, len(text), total_bytes, duration,
        )

    async def list_voices(self) -> list[VoiceInfo]:
        """Return the 6 built-in OpenAI TTS voices."""
        return list(self.VOICES.values())

    async def is_available(self) -> bool:
        """Check if OPENAI_API_KEY is set."""
        return bool(os.environ.get("OPENAI_API_KEY"))
