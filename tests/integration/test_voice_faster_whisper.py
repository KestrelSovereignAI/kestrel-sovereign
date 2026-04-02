"""
Integration tests for faster-whisper STT — real local transcription.

Requires faster-whisper package. Uses the 'tiny' model for speed.
Skipped automatically when faster-whisper is not installed.
"""
import io
import math
import os
import struct
import wave
import pytest

try:
    import faster_whisper  # noqa: F401
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

WHISPER_DATA_DIR = os.environ.get("WHISPER_DATA_DIR", "/Volumes/data2/models/whisper")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not FASTER_WHISPER_AVAILABLE,
        reason="faster-whisper not installed",
    ),
]


def _generate_sine_wav(frequency=440, duration=1.0, sample_rate=16000) -> bytes:
    """Generate a sine wave WAV for format testing."""
    samples = int(sample_rate * duration)
    data = []
    for i in range(samples):
        sample = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * i / sample_rate))
        data.append(struct.pack("<h", sample))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"".join(data))
    return buf.getvalue()


def _generate_speech_wav() -> bytes | None:
    """Generate speech WAV via Piper if available, else return None."""
    try:
        from kestrel_sovereign.voice.piper_tts import PiperTTSProvider
        from pathlib import Path

        data_dir = os.environ.get("PIPER_DATA_DIR", "/Volumes/data2/models/piper")
        if not Path(data_dir).exists() or not any(Path(data_dir).glob("*.onnx")):
            return None

        import asyncio
        provider = PiperTTSProvider({"data_dir": data_dir})

        async def _synth():
            return await provider.synthesize(
                "The quick brown fox jumps over the lazy dog",
                voice_id="", output_format="wav",
            )

        return asyncio.get_event_loop().run_until_complete(_synth())
    except Exception:
        return None


@pytest.fixture
def provider():
    from kestrel_sovereign.voice.faster_whisper_stt import FasterWhisperSTTProvider
    return FasterWhisperSTTProvider(config={
        "model": "tiny",  # Smallest model for fast tests
        "device": "auto",
        "data_dir": WHISPER_DATA_DIR,
    })


class TestFasterWhisperReal:
    """Real faster-whisper transcription on local hardware."""

    @pytest.mark.asyncio
    async def test_is_available(self, provider):
        assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_transcribe_silence_returns_empty_or_short(self, provider):
        """Transcribing a tone (no speech) should return empty or very short text."""
        wav = _generate_sine_wav(frequency=440, duration=1.0)
        result = await provider.transcribe(wav, audio_format="wav")
        # Whisper might hallucinate a few words on silence/tone, but not much
        assert len(result.split()) < 10

    @pytest.mark.asyncio
    async def test_transcribe_real_speech(self, provider):
        """If Piper is available, generate speech and transcribe it."""
        speech_wav = _generate_speech_wav()
        if speech_wav is None:
            pytest.skip("Piper not available for speech generation")

        result = await provider.transcribe(speech_wav, audio_format="wav")
        result_lower = result.lower()
        # Should get at least some words right
        assert any(word in result_lower for word in ["fox", "dog", "brown", "quick", "lazy"]), \
            f"Transcription didn't match expected content: {result}"

    @pytest.mark.asyncio
    async def test_device_detection(self, provider):
        """Verify device auto-detection returns a valid device."""
        # The provider should have resolved device during model load
        assert provider._device in ("cpu", "cuda", "auto")
