"""
Faster-Whisper STT Provider.

Local, privacy-safe speech-to-text using CTranslate2-based Whisper.
Runs 4x faster than OpenAI Whisper on CPU, supports CUDA/MPS acceleration.
"""
import asyncio
import io
import logging
import tempfile
from typing import AsyncIterator

from .base import STTProvider

logger = logging.getLogger(__name__)

# Supported models and devices
MODELS = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]
DEVICE_OPTIONS = ["auto", "cpu", "cuda", "mps"]


def _detect_device() -> str:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _compute_type_for_device(device: str) -> str:
    """Select optimal compute type for the given device."""
    if device == "cuda":
        return "float16"
    return "int8"


class FasterWhisperSTTProvider(STTProvider):
    """STT provider using faster-whisper for local, privacy-safe transcription."""

    name = "faster_whisper"
    is_local = True

    def __init__(self, config: dict | None = None):
        """Initialize the faster-whisper STT provider.

        Args:
            config: Configuration from [voice.faster_whisper] section.
        """
        config = config or {}
        self._model_name = config.get("model", "large-v3")
        self._device_config = config.get("device", "auto")
        self._data_dir = config.get("data_dir", "/data/models/whisper")
        self._compute_type_config = config.get("compute_type", "auto")
        self._model = None
        self._model_lock = asyncio.Lock()

    @property
    def device(self) -> str:
        """Resolve the actual device to use."""
        if self._device_config == "auto":
            return _detect_device()
        return self._device_config

    @property
    def compute_type(self) -> str:
        """Resolve the actual compute type to use."""
        if self._compute_type_config == "auto":
            return _compute_type_for_device(self.device)
        return self._compute_type_config

    def _load_model(self):
        """Load the WhisperModel (synchronous, called via to_thread)."""
        from faster_whisper import WhisperModel

        device = self.device
        # faster-whisper doesn't support MPS directly; fall back to CPU
        if device == "mps":
            device = "cpu"

        return WhisperModel(
            self._model_name,
            device=device,
            compute_type=self.compute_type,
            download_root=self._data_dir,
        )

    async def _get_model(self):
        """Get or create the cached WhisperModel instance."""
        if self._model is None:
            async with self._model_lock:
                if self._model is None:
                    self._model = await asyncio.to_thread(self._load_model)
        return self._model

    async def transcribe(self, audio: bytes, language: str = "",
                         audio_format: str = "opus") -> str:
        """Transcribe audio bytes to text using faster-whisper.

        Args:
            audio: Audio data as bytes.
            language: Optional ISO 639-1 language hint.
            audio_format: Audio format (opus, mp3, pcm, wav).

        Returns:
            Transcribed text.
        """
        model = await self._get_model()

        def _run_transcription():
            # Write audio to a temp file for faster-whisper
            suffix = f".{audio_format}" if audio_format else ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(audio)
                tmp.flush()

                kwargs = {}
                if language:
                    kwargs["language"] = language

                segments, _info = model.transcribe(tmp.name, **kwargs)
                return " ".join(segment.text.strip() for segment in segments)

        return await asyncio.to_thread(_run_transcription)

    async def transcribe_stream(self, audio_stream: AsyncIterator[bytes],
                                language: str = "") -> AsyncIterator[str]:
        """Stream transcription from audio chunks.

        Accumulates audio chunks and runs transcription periodically,
        yielding partial transcripts as segments complete.

        Args:
            audio_stream: Async iterator of audio chunks.
            language: Optional ISO 639-1 language hint.

        Yields:
            Transcribed text segments.
        """
        model = await self._get_model()
        buffer = io.BytesIO()

        async for chunk in audio_stream:
            buffer.write(chunk)

            # Process when we have enough data (~32KB chunks)
            if buffer.tell() >= 32768:
                audio_data = buffer.getvalue()
                buffer = io.BytesIO()

                def _run_segment(data=audio_data):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                        tmp.write(data)
                        tmp.flush()

                        kwargs = {}
                        if language:
                            kwargs["language"] = language

                        segments, _info = model.transcribe(tmp.name, **kwargs)
                        return [segment.text.strip() for segment in segments]

                texts = await asyncio.to_thread(_run_segment)
                for text in texts:
                    if text:
                        yield text

        # Process remaining audio in buffer
        remaining = buffer.getvalue()
        if remaining:
            def _run_remaining():
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    tmp.write(remaining)
                    tmp.flush()

                    kwargs = {}
                    if language:
                        kwargs["language"] = language

                    segments, _info = model.transcribe(tmp.name, **kwargs)
                    return [segment.text.strip() for segment in segments]

            texts = await asyncio.to_thread(_run_remaining)
            for text in texts:
                if text:
                    yield text

    async def is_available(self) -> bool:
        """Check if faster-whisper package is installed."""
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False
