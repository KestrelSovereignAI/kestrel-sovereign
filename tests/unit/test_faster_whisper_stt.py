"""Unit tests for FasterWhisperSTTProvider."""
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from kestrel_sovereign.voice.faster_whisper_stt import (
    FasterWhisperSTTProvider,
    _detect_device,
    _compute_type_for_device,
)


class TestDeviceDetection:
    """Tests for device auto-detection logic."""

    def test_detect_device_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _detect_device() == "cuda"

    def test_detect_device_mps(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _detect_device() == "mps"

    def test_detect_device_cpu_fallback(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _detect_device() == "cpu"

    def test_detect_device_no_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            assert _detect_device() == "cpu"

    def test_compute_type_cuda(self):
        assert _compute_type_for_device("cuda") == "float16"

    def test_compute_type_cpu(self):
        assert _compute_type_for_device("cpu") == "int8"

    def test_compute_type_mps(self):
        assert _compute_type_for_device("mps") == "int8"


class TestFasterWhisperSTTProvider:
    """Tests for the FasterWhisperSTTProvider class."""

    def test_default_config(self):
        provider = FasterWhisperSTTProvider()
        assert provider.name == "faster_whisper"
        assert provider.is_local is True
        assert provider._model_name == "large-v3"
        assert provider._device_config == "auto"
        assert provider._data_dir == "/data/models/whisper"
        assert provider._compute_type_config == "auto"

    def test_custom_config(self):
        config = {
            "model": "small",
            "device": "cuda",
            "data_dir": "/custom/path",
            "compute_type": "float16",
        }
        provider = FasterWhisperSTTProvider(config=config)
        assert provider._model_name == "small"
        assert provider._device_config == "cuda"
        assert provider._data_dir == "/custom/path"
        assert provider._compute_type_config == "float16"

    def test_device_property_explicit(self):
        provider = FasterWhisperSTTProvider(config={"device": "cuda"})
        assert provider.device == "cuda"

    def test_device_property_auto(self):
        provider = FasterWhisperSTTProvider(config={"device": "auto"})
        with patch("kestrel_sovereign.voice.faster_whisper_stt._detect_device", return_value="cpu"):
            assert provider.device == "cpu"

    def test_compute_type_explicit(self):
        provider = FasterWhisperSTTProvider(config={"compute_type": "float16"})
        assert provider.compute_type == "float16"

    def test_compute_type_auto_resolves(self):
        provider = FasterWhisperSTTProvider(config={"device": "cuda", "compute_type": "auto"})
        assert provider.compute_type == "float16"


class TestAvailability:
    """Tests for is_available check."""

    @pytest.mark.asyncio
    async def test_available_when_installed(self):
        provider = FasterWhisperSTTProvider()
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"faster_whisper": mock_module}):
            assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_unavailable_when_not_installed(self):
        provider = FasterWhisperSTTProvider()
        with patch.dict("sys.modules", {"faster_whisper": None}):
            assert await provider.is_available() is False


class TestModelCaching:
    """Tests for model caching behavior."""

    @pytest.mark.asyncio
    async def test_model_loaded_once(self):
        provider = FasterWhisperSTTProvider(config={"device": "cpu"})
        mock_model = MagicMock()

        with patch.object(provider, "_load_model", return_value=mock_model) as load_mock:
            model1 = await provider._get_model()
            model2 = await provider._get_model()

            assert model1 is model2
            load_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_not_loaded_until_needed(self):
        provider = FasterWhisperSTTProvider()
        assert provider._model is None


class TestTranscription:
    """Tests for transcription with mocked WhisperModel."""

    @pytest.mark.asyncio
    async def test_transcribe_returns_text(self):
        provider = FasterWhisperSTTProvider(config={"device": "cpu"})

        # Mock segments returned by faster-whisper
        seg1 = MagicMock()
        seg1.text = " Hello world "
        seg2 = MagicMock()
        seg2.text = " How are you "

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())
        provider._model = mock_model

        result = await provider.transcribe(b"fake-audio-data", language="en")

        assert result == "Hello world How are you"
        mock_model.transcribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_stream_yields_segments(self):
        provider = FasterWhisperSTTProvider(config={"device": "cpu"})

        seg = MagicMock()
        seg.text = "streamed text"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], MagicMock())
        provider._model = mock_model

        async def audio_chunks():
            # Yield enough data to trigger processing (>32KB)
            yield b"\x00" * 33000
            yield b"\x00" * 100  # remaining buffer

        results = []
        async for text in provider.transcribe_stream(audio_chunks()):
            results.append(text)

        assert len(results) > 0
        assert "streamed text" in results

    @pytest.mark.asyncio
    async def test_load_model_mps_falls_back_to_cpu(self):
        provider = FasterWhisperSTTProvider(config={"device": "mps"})

        mock_whisper_model = MagicMock()
        with patch("kestrel_sovereign.voice.faster_whisper_stt.WhisperModel", mock_whisper_model, create=True):
            # Patch the import inside _load_model
            import kestrel_sovereign.voice.faster_whisper_stt as mod
            with patch.object(mod, "__import__", create=True):
                mock_cls = MagicMock()
                with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_cls)}):
                    # Re-call _load_model directly
                    from kestrel_sovereign.voice.faster_whisper_stt import FasterWhisperSTTProvider as Cls
                    p = Cls(config={"device": "mps"})

                    p._load_model()
                    mock_cls.assert_called_once_with(
                        "large-v3",
                        device="cpu",
                        compute_type="int8",
                        download_root="/data/models/whisper",
                    )
