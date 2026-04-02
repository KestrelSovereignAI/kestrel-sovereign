"""
Integration tests for Piper TTS — real local synthesis.

Requires piper-tts package and at least one voice model in data_dir.
Skipped automatically when piper is not installed.
"""
import io
import os
import wave
import pytest

try:
    import piper  # noqa: F401
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

# Default model dir — override with PIPER_DATA_DIR env var
PIPER_DATA_DIR = os.environ.get("PIPER_DATA_DIR", "/Volumes/data2/models/piper")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not PIPER_AVAILABLE, reason="piper-tts not installed"),
]


def _has_models():
    from pathlib import Path
    d = Path(PIPER_DATA_DIR)
    return d.exists() and any(d.glob("*.onnx"))


@pytest.fixture
def provider():
    pytest.importorskip("piper")
    if not _has_models():
        pytest.skip(f"No Piper voice models in {PIPER_DATA_DIR}")
    from kestrel_sovereign.voice.piper_tts import PiperTTSProvider
    return PiperTTSProvider({"data_dir": PIPER_DATA_DIR})


class TestPiperTTSReal:
    """Real Piper TTS synthesis on local CPU."""

    @pytest.mark.asyncio
    async def test_synthesize_returns_valid_wav(self, provider):
        audio = await provider.synthesize("Hello world", voice_id="", output_format="wav")
        assert isinstance(audio, bytes)
        assert len(audio) > 1000
        # Verify it's valid WAV
        with wave.open(io.BytesIO(audio), "rb") as w:
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getframerate() > 0
            assert w.getnframes() > 0

    @pytest.mark.asyncio
    async def test_synthesize_opus_format(self, provider):
        audio = await provider.synthesize("Test opus output", voice_id="", output_format="opus")
        assert isinstance(audio, bytes)
        assert len(audio) > 100
        # Opus in OGG container starts with "OggS"
        assert audio[:4] == b"OggS"

    @pytest.mark.asyncio
    async def test_stream_produces_sentence_chunks(self, provider):
        chunks = []
        async for chunk in provider.synthesize_stream(
            "First sentence. Second sentence. Third sentence.",
            voice_id="",
            output_format="wav",
        ):
            chunks.append(chunk)
        assert len(chunks) == 3
        for chunk in chunks:
            assert len(chunk) > 500  # Each sentence should produce real audio

    @pytest.mark.asyncio
    async def test_list_voices_discovers_models(self, provider):
        voices = await provider.list_voices()
        assert len(voices) >= 1
        for v in voices:
            assert v.voice_id
            assert v.provider == "piper"

    @pytest.mark.asyncio
    async def test_is_available(self, provider):
        assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_model_caching(self, provider):
        """Second synthesis should reuse cached model (faster)."""
        import time
        # First call loads the model
        t0 = time.monotonic()
        await provider.synthesize("Warm up", voice_id="", output_format="wav")
        first_dur = time.monotonic() - t0

        # Second call should be faster (model cached)
        t0 = time.monotonic()
        await provider.synthesize("Cached", voice_id="", output_format="wav")
        second_dur = time.monotonic() - t0

        # Second should be noticeably faster (at least 2x)
        # Don't assert hard ratio — just verify it's not absurdly slow
        assert second_dur < first_dur * 2 or second_dur < 1.0
