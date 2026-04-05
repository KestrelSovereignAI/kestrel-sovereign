"""
Deepgram STT Provider.

Real-time streaming transcription via Deepgram's WebSocket API,
plus batch transcription via REST. Cloud provider (requires API key).
"""
import asyncio
import logging
import os
from typing import AsyncIterator

from .base import STTProvider

logger = logging.getLogger(__name__)

# Deepgram SDK is optional — guarded at import time
try:
    from deepgram import DeepgramClient, DeepgramClientOptions, LiveOptions, PrerecordedOptions
    from deepgram.clients.listen.v1.websocket import AsyncListenWebSocketClient

    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False
    DeepgramClient = None  # type: ignore[assignment,misc]
    DeepgramClientOptions = None  # type: ignore[assignment,misc]
    LiveOptions = None  # type: ignore[assignment,misc]
    PrerecordedOptions = None  # type: ignore[assignment,misc]
    AsyncListenWebSocketClient = None  # type: ignore[assignment,misc]


class DeepgramSTTProvider(STTProvider):
    """Deepgram speech-to-text provider using nova-2 models.

    Supports both batch (REST) and real-time streaming (WebSocket) transcription.
    """

    name = "deepgram"
    is_local = False

    MODELS = ["nova-2", "nova-2-general", "nova-2-meeting", "nova-2-phonecall"]

    def __init__(self, config: dict | None = None):
        """Initialize with optional config from [voice.deepgram] section.

        Args:
            config: Dict with keys like 'model', 'language'.
        """
        config = config or {}
        self._model = config.get("model", "nova-2")
        self._language = config.get("language", "en")

    async def is_available(self) -> bool:
        """Check if deepgram-sdk is installed and DEEPGRAM_API_KEY is set."""
        if not DEEPGRAM_AVAILABLE:
            logger.debug("Deepgram SDK not installed")
            return False
        if not os.environ.get("DEEPGRAM_API_KEY"):
            logger.debug("DEEPGRAM_API_KEY not set")
            return False
        return True

    async def transcribe(self, audio: bytes, language: str = "",
                         audio_format: str = "opus") -> str:
        """Transcribe audio bytes via Deepgram REST API.

        Args:
            audio: Audio data as bytes.
            language: ISO 639-1 language hint (overrides default).
            audio_format: Audio format (opus, mp3, pcm, wav).

        Returns:
            Transcribed text.
        """
        if not DEEPGRAM_AVAILABLE:
            raise RuntimeError("deepgram-sdk is not installed")

        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY environment variable not set")

        client = DeepgramClient(api_key)
        lang = language or self._language

        mime_map = {
            "opus": "audio/ogg",
            "ogg": "audio/ogg",
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "pcm": "audio/l16",
            "webm": "audio/webm",
        }
        mimetype = mime_map.get(audio_format, "audio/wav")

        source = {"buffer": audio, "mimetype": mimetype}
        options = PrerecordedOptions(
            model=self._model,
            language=lang,
            punctuate=True,
            smart_format=True,
        )

        logger.info("Deepgram batch transcription: model=%s language=%s format=%s bytes=%d",
                     self._model, lang, audio_format, len(audio))

        response = await client.listen.asyncrest.v("1").transcribe_file(source, options)
        transcript = response.results.channels[0].alternatives[0].transcript

        word_count = len(transcript.split()) if transcript else 0
        logger.info("Deepgram transcription complete: words=%d", word_count)

        return transcript

    async def transcribe_stream(self, audio_stream: AsyncIterator[bytes],
                                language: str = "") -> AsyncIterator[str]:
        """Stream transcription via Deepgram WebSocket.

        Connects a WebSocket, feeds audio chunks, and yields partial/final
        transcripts as they arrive from Deepgram.

        Args:
            audio_stream: Async iterator of audio chunks.
            language: ISO 639-1 language hint (overrides default).

        Yields:
            Transcribed text segments (both interim and final).
        """
        if not DEEPGRAM_AVAILABLE:
            raise RuntimeError("deepgram-sdk is not installed")

        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY environment variable not set")

        lang = language or self._language
        transcript_queue: asyncio.Queue[str | None] = asyncio.Queue()

        client = DeepgramClient(api_key)
        connection: AsyncListenWebSocketClient = client.listen.asyncwebsocket.v("1")

        async def on_message(_connection, result, **kwargs):
            """Handle transcript results from Deepgram."""
            transcript = result.channel.alternatives[0].transcript
            if transcript:
                is_final = result.is_final
                logger.debug("Deepgram stream: transcript=%r final=%s", transcript, is_final)
                await transcript_queue.put(transcript)

        async def on_error(_connection, error, **kwargs):
            """Handle errors from Deepgram WebSocket."""
            logger.error("Deepgram WebSocket error: %s", error)
            await transcript_queue.put(None)

        connection.on("Results", on_message)
        connection.on("Error", on_error)

        options = LiveOptions(
            model=self._model,
            language=lang,
            punctuate=True,
            smart_format=True,
            interim_results=True,
            utterance_end_ms="1000",
        )

        logger.info("Deepgram streaming: model=%s language=%s", self._model, lang)

        started = await connection.start(options)
        if not started:
            raise RuntimeError("Failed to start Deepgram WebSocket connection")

        async def _send_audio():
            """Send audio chunks then signal end of stream."""
            try:
                async for chunk in audio_stream:
                    await connection.send(chunk)
                await connection.finish()
            except Exception as e:
                logger.error("Error sending audio to Deepgram: %s", e)
                await transcript_queue.put(None)

        send_task = asyncio.create_task(_send_audio())

        try:
            while True:
                transcript = await transcript_queue.get()
                if transcript is None:
                    break
                yield transcript
        finally:
            send_task.cancel()
            try:
                await send_task
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                pass
            except Exception as e:
                logger.error("Unexpected error in Deepgram send_task cleanup: %s", e)
            try:
                await connection.finish()
            except Exception:
                pass
