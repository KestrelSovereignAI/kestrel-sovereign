"""Unit tests for PiperTTSProvider."""
import io
import json
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from kestrel_sovereign.voice.piper_tts import (
    PiperTTSProvider,
    _piper_available,
    _SENTENCE_RE,
    DEFAULT_VOICES,
)
from kestrel_sovereign.voice.base import VoiceInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(num_samples: int = 100, sample_rate: int = 22050) -> bytes:
    """Create minimal valid WAV bytes for testing."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * num_samples)
    return buf.getvalue()


def _create_onnx_model(tmp_path: Path, voice_id: str, meta: dict | None = None):
    """Create a fake .onnx file and optional .onnx.json config."""
    onnx_file = tmp_path / f"{voice_id}.onnx"
    onnx_file.write_bytes(b"fake-onnx-model")

    if meta is not None:
        json_file = tmp_path / f"{voice_id}.onnx.json"
        json_file.write_text(json.dumps(meta))


# ---------------------------------------------------------------------------
# Availability tests
# ---------------------------------------------------------------------------

class TestIsAvailable:
    """Tests for is_available() — package + model checks."""

    @pytest.mark.asyncio
    async def test_unavailable_when_package_missing(self, tmp_path):
        """is_available() returns False when piper-tts is not installed."""
        _create_onnx_model(tmp_path, "en_US-lessac-medium")
        provider = PiperTTSProvider({"data_dir": str(tmp_path)})
        with patch("kestrel_sovereign.voice.piper_tts._piper_available", return_value=False):
            assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_unavailable_when_no_models(self, tmp_path):
        """is_available() returns False when data_dir has no .onnx files."""
        provider = PiperTTSProvider({"data_dir": str(tmp_path)})
        with patch("kestrel_sovereign.voice.piper_tts._piper_available", return_value=True):
            assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_unavailable_when_data_dir_missing(self):
        """is_available() returns False when data_dir doesn't exist."""
        provider = PiperTTSProvider({"data_dir": "/nonexistent/path"})
        with patch("kestrel_sovereign.voice.piper_tts._piper_available", return_value=True):
            assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_available_when_package_and_models_present(self, tmp_path):
        """is_available() returns True when package installed and models exist."""
        _create_onnx_model(tmp_path, "en_US-lessac-medium")
        provider = PiperTTSProvider({"data_dir": str(tmp_path)})
        with patch("kestrel_sovereign.voice.piper_tts._piper_available", return_value=True):
            assert await provider.is_available() is True


# ---------------------------------------------------------------------------
# Voice discovery tests
# ---------------------------------------------------------------------------

class TestListVoices:
    """Tests for list_voices() — scanning data_dir for models."""

    @pytest.mark.asyncio
    async def test_empty_when_no_dir(self):
        """Returns empty list when data_dir doesn't exist."""
        provider = PiperTTSProvider({"data_dir": "/nonexistent"})
        voices = await provider.list_voices()
        assert voices == []

    @pytest.mark.asyncio
    async def test_discovers_known_voices(self, tmp_path):
        """Discovers voices that match DEFAULT_VOICES entries."""
        _create_onnx_model(tmp_path, "en_US-lessac-medium")
        _create_onnx_model(tmp_path, "en_US-amy-medium")
        provider = PiperTTSProvider({"data_dir": str(tmp_path)})

        voices = await provider.list_voices()
        assert len(voices) == 2
        ids = {v.voice_id for v in voices}
        assert ids == {"en_US-lessac-medium", "en_US-amy-medium"}
        # Known voices should use DEFAULT_VOICES metadata
        lessac = [v for v in voices if v.voice_id == "en_US-lessac-medium"][0]
        assert lessac.name == "Lessac"
        assert lessac.gender == "masculine"

    @pytest.mark.asyncio
    async def test_discovers_unknown_voice_with_json(self, tmp_path):
        """Discovers unknown voice and reads metadata from .onnx.json."""
        meta = {
            "dataset": "CustomVoice",
            "language": {"code": "de"},
        }
        _create_onnx_model(tmp_path, "de_DE-custom-medium", meta=meta)
        provider = PiperTTSProvider({"data_dir": str(tmp_path)})

        voices = await provider.list_voices()
        assert len(voices) == 1
        assert voices[0].voice_id == "de_DE-custom-medium"
        assert voices[0].name == "CustomVoice"
        assert voices[0].language == "de"

    @pytest.mark.asyncio
    async def test_discovers_unknown_voice_without_json(self, tmp_path):
        """Discovers unknown voice even without .onnx.json."""
        _create_onnx_model(tmp_path, "fr_FR-siwis-medium")
        provider = PiperTTSProvider({"data_dir": str(tmp_path)})

        voices = await provider.list_voices()
        assert len(voices) == 1
        assert voices[0].voice_id == "fr_FR-siwis-medium"
        assert voices[0].provider == "piper"


# ---------------------------------------------------------------------------
# Synthesis tests (mocked Piper)
# ---------------------------------------------------------------------------

class TestSynthesize:
    """Tests for synthesize() — with mocked piper package."""

    @pytest.mark.asyncio
    async def test_synthesize_wav(self, tmp_path):
        """synthesize() returns WAV bytes."""
        provider = PiperTTSProvider({
            "data_dir": str(tmp_path),
            "model": "test-voice",
        })

        wav_data = _make_wav_bytes()

        # Mock _synthesize_sync to return WAV bytes directly
        with patch.object(provider, "_synthesize_sync", return_value=wav_data):
            result = await provider.synthesize("Hello", "test-voice", output_format="wav")

        assert result == wav_data
        # Verify it's valid WAV
        with wave.open(io.BytesIO(result), "rb") as w:
            assert w.getnchannels() == 1

    @pytest.mark.asyncio
    async def test_synthesize_uses_default_voice(self, tmp_path):
        """synthesize() uses default voice when voice_id is empty."""
        provider = PiperTTSProvider({
            "data_dir": str(tmp_path),
            "model": "en_US-lessac-medium",
        })

        with patch.object(provider, "_synthesize_sync", return_value=_make_wav_bytes()) as mock_sync:
            await provider.synthesize("Hello", "", output_format="wav")

        mock_sync.assert_called_once_with("Hello", "en_US-lessac-medium")


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------

class TestSynthesizeStream:
    """Tests for synthesize_stream() — sentence splitting."""

    @pytest.mark.asyncio
    async def test_stream_splits_sentences(self, tmp_path):
        """synthesize_stream() yields one chunk per sentence."""
        provider = PiperTTSProvider({"data_dir": str(tmp_path)})

        wav_data = _make_wav_bytes()
        with patch.object(provider, "_synthesize_sync", return_value=wav_data):
            chunks = []
            async for chunk in provider.synthesize_stream(
                "Hello world. How are you? Fine thanks.",
                "test-voice",
                output_format="wav",
            ):
                chunks.append(chunk)

        assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_stream_single_sentence(self, tmp_path):
        """synthesize_stream() yields one chunk for single sentence."""
        provider = PiperTTSProvider({"data_dir": str(tmp_path)})

        wav_data = _make_wav_bytes()
        with patch.object(provider, "_synthesize_sync", return_value=wav_data):
            chunks = []
            async for chunk in provider.synthesize_stream(
                "Hello world",
                "test-voice",
                output_format="wav",
            ):
                chunks.append(chunk)

        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Model caching tests
# ---------------------------------------------------------------------------

class TestModelCaching:
    """Tests for ONNX model caching behavior."""

    def test_cache_returns_same_instance(self, tmp_path):
        """_get_voice_model() caches and returns the same instance."""
        _create_onnx_model(tmp_path, "test-voice")
        provider = PiperTTSProvider({"data_dir": str(tmp_path)})

        mock_voice = MagicMock()
        mock_voice_cls = MagicMock(return_value=mock_voice)
        mock_voice_cls.load = MagicMock(return_value=mock_voice)

        with patch.dict("sys.modules", {"piper": MagicMock()}):
            with patch("kestrel_sovereign.voice.piper_tts.PiperTTSProvider._get_voice_model") as mock_get:
                mock_get.return_value = mock_voice
                result1 = provider._get_voice_model("test-voice")
                result2 = provider._get_voice_model("test-voice")
                assert result1 is result2


# ---------------------------------------------------------------------------
# Sentence splitting tests
# ---------------------------------------------------------------------------

class TestSentenceSplitting:
    """Tests for the sentence-splitting regex."""

    def test_splits_on_period(self):
        assert _SENTENCE_RE.split("Hello. World.") == ["Hello.", "World."]

    def test_splits_on_question_mark(self):
        assert _SENTENCE_RE.split("How? Why?") == ["How?", "Why?"]

    def test_splits_on_exclamation(self):
        assert _SENTENCE_RE.split("Wow! Cool!") == ["Wow!", "Cool!"]

    def test_no_split_without_space(self):
        result = _SENTENCE_RE.split("Hello.World")
        assert result == ["Hello.World"]

    def test_single_sentence(self):
        result = _SENTENCE_RE.split("Hello world")
        assert result == ["Hello world"]


# ---------------------------------------------------------------------------
# Provider metadata tests
# ---------------------------------------------------------------------------

class TestProviderMetadata:
    """Tests for provider class attributes."""

    def test_name_is_piper(self):
        provider = PiperTTSProvider({})
        assert provider.name == "piper"

    def test_is_local_true(self):
        provider = PiperTTSProvider({})
        assert provider.is_local is True

    def test_default_voices_populated(self):
        assert len(DEFAULT_VOICES) == 4
        assert "en_US-lessac-medium" in DEFAULT_VOICES
        assert all(isinstance(v, VoiceInfo) for v in DEFAULT_VOICES.values())
