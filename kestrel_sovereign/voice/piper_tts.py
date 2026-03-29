"""
Piper TTS Provider — local, CPU-based, privacy-safe text-to-speech.

Uses the piper-tts Python package with ONNX voice models for fully offline
synthesis. No cloud calls, suitable for EPHEMERAL and ISOLATED privacy modes.
"""
import asyncio
import io
import json
import logging
import re
import threading
from pathlib import Path
from typing import AsyncIterator, Optional

from .base import TTSProvider, VoiceInfo

logger = logging.getLogger(__name__)

# Sentence-splitting regex: split on sentence-ending punctuation followed by whitespace
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

# Default voice models (well-known English voices)
DEFAULT_VOICES = {
    "en_US-lessac-medium": VoiceInfo(
        voice_id="en_US-lessac-medium", name="Lessac",
        provider="piper", gender="masculine",
        age="middle", energy="calm", accent="american",
    ),
    "en_US-amy-medium": VoiceInfo(
        voice_id="en_US-amy-medium", name="Amy",
        provider="piper", gender="feminine",
        age="middle", energy="warm", accent="american",
    ),
    "en_US-ryan-medium": VoiceInfo(
        voice_id="en_US-ryan-medium", name="Ryan",
        provider="piper", gender="masculine",
        age="young", energy="energetic", accent="american",
    ),
    "en_GB-alba-medium": VoiceInfo(
        voice_id="en_GB-alba-medium", name="Alba",
        provider="piper", gender="feminine",
        age="young", energy="warm", accent="british",
    ),
}


def _piper_available() -> bool:
    """Check if the piper-tts package is importable."""
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        return False


def _soundfile_available() -> bool:
    """Check if soundfile is importable."""
    try:
        import soundfile  # noqa: F401
        return True
    except ImportError:
        return False


class PiperTTSProvider(TTSProvider):
    """Local TTS using Piper (ONNX-based, CPU, no cloud calls)."""

    name = "piper"
    is_local = True

    def __init__(self, config: dict):
        """Initialize with voice config from [voice.piper] section.

        Args:
            config: Dict with keys like 'model' (default voice) and
                    'data_dir' (path to ONNX voice models).
        """
        self._default_voice = config.get("model", "en_US-lessac-medium")
        self._data_dir = Path(config.get("data_dir", "/data/models/piper"))
        # Cache: voice_id -> PiperVoice instance
        self._model_cache: dict = {}
        self._cache_lock = threading.Lock()

    def _get_voice_model(self, voice_id: str):
        """Load or retrieve a cached Piper voice model (not async — called via to_thread).

        Args:
            voice_id: Voice model name (e.g. "en_US-lessac-medium").

        Returns:
            A piper.PiperVoice instance.

        Raises:
            FileNotFoundError: If the ONNX model file doesn't exist.
            ImportError: If piper-tts isn't installed.
        """
        with self._cache_lock:
            if voice_id in self._model_cache:
                return self._model_cache[voice_id]

        from piper import PiperVoice

        model_path = self._data_dir / f"{voice_id}.onnx"
        config_path = self._data_dir / f"{voice_id}.onnx.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper voice model not found: {model_path}. "
                f"Download it to {self._data_dir}/"
            )

        config_path_arg = str(config_path) if config_path.exists() else None
        voice = PiperVoice.load(str(model_path), config_path=config_path_arg)

        with self._cache_lock:
            self._model_cache[voice_id] = voice

        return voice

    def _synthesize_sync(self, text: str, voice_id: str) -> bytes:
        """Synchronous synthesis — runs in a thread via asyncio.to_thread().

        Returns raw WAV bytes (16-bit PCM).
        """
        voice = self._get_voice_model(voice_id)
        wav_buf = io.BytesIO()
        with wave_file(wav_buf, voice.config) as wav:
            voice.synthesize(text, wav)
        return wav_buf.getvalue()

    def _convert_format(self, wav_bytes: bytes, output_format: str) -> bytes:
        """Convert WAV bytes to the requested output format using soundfile.

        Args:
            wav_bytes: Raw WAV data.
            output_format: Target format (wav, mp3, opus, ogg, flac).

        Returns:
            Converted audio bytes.
        """
        if output_format == "wav":
            return wav_bytes

        import soundfile as sf
        import numpy as np

        # Read the WAV bytes
        wav_buf = io.BytesIO(wav_bytes)
        data, samplerate = sf.read(wav_buf)

        # Map output format to soundfile format/subtype
        format_map = {
            "ogg": "OGG",
            "opus": "OGG",
            "flac": "FLAC",
        }
        sf_format = format_map.get(output_format)
        if sf_format is None:
            logger.warning(
                f"Unsupported output format '{output_format}', returning WAV"
            )
            return wav_bytes

        subtype_map = {
            "ogg": "VORBIS",
            "opus": "OPUS",
            "flac": "PCM_16",
        }

        out_buf = io.BytesIO()
        sf.write(
            out_buf, data, samplerate,
            format=sf_format,
            subtype=subtype_map.get(output_format, "PCM_16"),
        )
        return out_buf.getvalue()

    async def synthesize(self, text: str, voice_id: str, model: str = "",
                         output_format: str = "opus") -> bytes:
        """Synthesize text to audio bytes using Piper.

        CPU-bound work is offloaded via asyncio.to_thread().
        """
        vid = voice_id or self._default_voice
        wav_bytes = await asyncio.to_thread(self._synthesize_sync, text, vid)

        if output_format != "wav":
            wav_bytes = await asyncio.to_thread(
                self._convert_format, wav_bytes, output_format
            )
        return wav_bytes

    async def synthesize_stream(self, text: str, voice_id: str, model: str = "",
                                output_format: str = "opus") -> AsyncIterator[bytes]:
        """Stream audio sentence-by-sentence.

        Splits input on sentence boundaries and yields one audio chunk per sentence.
        """
        vid = voice_id or self._default_voice
        sentences = _SENTENCE_RE.split(text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            wav_bytes = await asyncio.to_thread(
                self._synthesize_sync, sentence, vid
            )
            if output_format != "wav":
                wav_bytes = await asyncio.to_thread(
                    self._convert_format, wav_bytes, output_format
                )
            yield wav_bytes

    async def list_voices(self) -> list[VoiceInfo]:
        """Discover installed voice models by scanning data_dir for .onnx files."""
        voices: list[VoiceInfo] = []

        if not self._data_dir.exists():
            return voices

        for onnx_path in sorted(self._data_dir.glob("*.onnx")):
            voice_id = onnx_path.stem  # e.g. "en_US-lessac-medium"

            # Use defaults if we know this voice, otherwise build from filename
            if voice_id in DEFAULT_VOICES:
                voices.append(DEFAULT_VOICES[voice_id])
                continue

            # Try to extract metadata from the .onnx.json config
            json_path = onnx_path.with_suffix(".onnx.json")
            name = voice_id
            language = "en"
            gender = "neutral"

            if json_path.exists():
                try:
                    meta = json.loads(json_path.read_text())
                    dataset = meta.get("dataset", "")
                    if dataset:
                        name = dataset
                    lang_obj = meta.get("language", {})
                    if isinstance(lang_obj, dict):
                        language = lang_obj.get("code", language)
                    elif isinstance(lang_obj, str):
                        language = lang_obj
                    speaker = meta.get("speaker_id_map", {})
                    # Heuristic: if only one speaker, use its name
                    if len(speaker) == 1:
                        name = list(speaker.keys())[0]
                except (json.JSONDecodeError, OSError):
                    pass

            voices.append(VoiceInfo(
                voice_id=voice_id,
                name=name,
                provider="piper",
                language=language,
                gender=gender,
            ))

        return voices

    async def is_available(self) -> bool:
        """Check if piper-tts is installed AND at least one voice model exists."""
        if not _piper_available():
            logger.debug("piper-tts package not installed")
            return False

        if not self._data_dir.exists():
            logger.debug(f"Piper data_dir does not exist: {self._data_dir}")
            return False

        has_models = any(self._data_dir.glob("*.onnx"))
        if not has_models:
            logger.debug(f"No .onnx voice models in {self._data_dir}")
            return False

        return True


def wave_file(wav_buf: io.BytesIO, config):
    """Context manager that creates a WAV file writer using Piper's config.

    This mirrors piper's own wave file helper.
    """
    import wave

    wav = wave.open(wav_buf, "wb")
    wav.setnchannels(1)
    wav.setsampwidth(2)  # 16-bit
    wav.setframerate(config.sample_rate)

    class _WavContext:
        def __enter__(self):
            return wav
        def __exit__(self, *args):
            wav.close()

    return _WavContext()
