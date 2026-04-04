"""
OpenAI STT Provider — Whisper API adapter.

Wraps the OpenAI Whisper API for speech-to-text transcription.
"""
import io
import logging
import os
import time
from typing import AsyncIterator

import openai

from kestrel_sdk.voice.base import STTProvider

logger = logging.getLogger(__name__)

# Whisper API file size limit (25 MB)
MAX_FILE_SIZE = 25 * 1024 * 1024

SUPPORTED_FORMATS = [
    "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm", "opus",
]

# Map audio_format to a filename extension the OpenAI API recognises
_FORMAT_TO_EXT = {
    "flac": "flac",
    "mp3": "mp3",
    "mp4": "mp4",
    "mpeg": "mpeg",
    "mpga": "mpga",
    "m4a": "m4a",
    "ogg": "ogg",
    "wav": "wav",
    "webm": "webm",
    "opus": "ogg",  # opus is typically in an ogg container
}


class OpenAISTTProvider(STTProvider):
    """OpenAI Whisper STT provider."""

    name = "openai"
    is_local = False

    MODELS = ["whisper-1"]

    def __init__(self, config: dict | None = None):
        """Initialise with optional voice config section.

        Args:
            config: The ``[voice.openai]`` section from kestrel.toml (or empty dict).
        """
        config = config or {}
        self._model = config.get("stt_model", "whisper-1")
        api_key = os.environ.get("OPENAI_API_KEY")
        self._client = openai.AsyncOpenAI(api_key=api_key) if api_key else None

    # ------------------------------------------------------------------
    # STTProvider ABC
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio: bytes,
        language: str = "",
        audio_format: str = "opus",
    ) -> str:
        """Transcribe audio bytes to text via the Whisper API.

        If *audio* exceeds 25 MB it is split into chunks and the
        transcriptions are concatenated.
        """
        if audio_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{audio_format}'. "
                f"Supported: {SUPPORTED_FORMATS}"
            )

        if not self._client:
            raise RuntimeError("OpenAI API key not configured for STT")

        client = self._client

        if len(audio) <= MAX_FILE_SIZE:
            return await self._transcribe_chunk(client, audio, language, audio_format)

        # Split into <=25 MB chunks and concatenate results
        logger.info(
            "Audio size %d bytes exceeds 25 MB limit — splitting into chunks",
            len(audio),
        )
        parts: list[str] = []
        for offset in range(0, len(audio), MAX_FILE_SIZE):
            chunk = audio[offset : offset + MAX_FILE_SIZE]
            text = await self._transcribe_chunk(client, chunk, language, audio_format)
            parts.append(text)
        return " ".join(parts)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        language: str = "",
    ) -> AsyncIterator[str]:
        """Yield partial transcripts by buffering audio and periodically transcribing.

        OpenAI Whisper does not natively support streaming partial
        transcripts, so we accumulate audio and call ``transcribe`` on the
        buffer at regular intervals (every ~1 MB of audio received).
        """
        if not self._client:
            raise RuntimeError("OpenAI API key not configured for STT")

        client = self._client

        buffer = bytearray()
        flush_threshold = 1 * 1024 * 1024  # 1 MB
        prev_text = ""

        async for chunk in audio_stream:
            buffer.extend(chunk)
            if len(buffer) >= flush_threshold:
                text = await self._transcribe_chunk(
                    client, bytes(buffer), language, "opus",
                )
                if text and text != prev_text:
                    yield text
                    prev_text = text

        # Final flush for any remaining audio
        if buffer:
            text = await self._transcribe_chunk(
                client, bytes(buffer), language, "opus",
            )
            if text and text != prev_text:
                yield text

    async def is_available(self) -> bool:
        """Return True when OPENAI_API_KEY is set."""
        return bool(os.environ.get("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _transcribe_chunk(
        self,
        client,
        audio: bytes,
        language: str,
        audio_format: str,
    ) -> str:
        """Call the Whisper API for a single audio chunk."""
        ext = _FORMAT_TO_EXT.get(audio_format, audio_format)
        file_tuple = (f"audio.{ext}", io.BytesIO(audio))

        start = time.monotonic()

        kwargs: dict = {
            "model": self._model,
            "file": file_tuple,
            "response_format": "text",
        }
        if language:
            kwargs["language"] = language

        result = await client.audio.transcriptions.create(**kwargs)
        elapsed = time.monotonic() - start

        # The text response_format returns a plain string
        text = result if isinstance(result, str) else str(result)

        logger.info(
            "Whisper transcription: model=%s audio_size=%d language=%s duration=%.2fs",
            self._model,
            len(audio),
            language or "auto",
            elapsed,
        )
        return text.strip()
