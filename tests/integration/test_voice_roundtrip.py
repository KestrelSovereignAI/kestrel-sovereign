"""
Integration tests for voice round-trips — TTS → STT across provider combinations.

Tests real audio flowing through the full pipeline.
"""
import os
import pytest

pytestmark = [pytest.mark.integration]


class TestOpenAIRoundTrip:
    """OpenAI TTS → OpenAI STT round-trip."""

    pytestmark = [
        pytest.mark.skipif(
            not os.environ.get("OPENAI_API_KEY"),
            reason="OPENAI_API_KEY not set",
        ),
    ]

    @pytest.mark.asyncio
    async def test_tts_stt_round_trip_all_formats(self):
        """Synthesize in each format, transcribe back, verify text matches."""
        from kestrel_sovereign.voice.openai_tts import OpenAITTSProvider
        from kestrel_sovereign.voice.openai_stt import OpenAISTTProvider

        tts = OpenAITTSProvider(config={"tts_model": "tts-1"})
        stt = OpenAISTTProvider(config={"stt_model": "whisper-1"})

        original_text = "Kestrel agents can now speak and listen"

        for fmt in ("mp3", "opus", "wav"):
            audio = await tts.synthesize(original_text, voice_id="nova", output_format=fmt)
            assert len(audio) > 100, f"Empty audio for format {fmt}"

            transcript = await stt.transcribe(audio, audio_format=fmt)
            transcript_lower = transcript.lower()
            assert "kestrel" in transcript_lower or "speak" in transcript_lower, \
                f"Round-trip failed for {fmt}: got '{transcript}'"


class TestPiperToWhisperRoundTrip:
    """Local-only round-trip: Piper TTS → faster-whisper STT."""

    @pytest.mark.asyncio
    async def test_local_round_trip(self):
        """Fully local voice pipeline — no cloud calls."""
        try:
            import piper  # noqa: F401
        except ImportError:
            pytest.skip("piper-tts not installed")
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            pytest.skip("faster-whisper not installed")

        from pathlib import Path
        piper_dir = os.environ.get("PIPER_DATA_DIR", "/Volumes/data2/models/piper")
        if not Path(piper_dir).exists() or not any(Path(piper_dir).glob("*.onnx")):
            pytest.skip(f"No Piper models in {piper_dir}")

        from kestrel_sovereign.voice.piper_tts import PiperTTSProvider
        from kestrel_sovereign.voice.faster_whisper_stt import FasterWhisperSTTProvider

        tts = PiperTTSProvider({"data_dir": piper_dir})
        stt = FasterWhisperSTTProvider(config={
            "model": "tiny",
            "data_dir": os.environ.get("WHISPER_DATA_DIR", "/Volumes/data2/models/whisper"),
        })

        # Synthesize speech locally
        audio = await tts.synthesize(
            "The weather is nice today",
            voice_id="", output_format="wav",
        )
        assert len(audio) > 1000

        # Transcribe locally
        transcript = await stt.transcribe(audio, audio_format="wav")
        transcript_lower = transcript.lower()
        assert any(word in transcript_lower for word in ["weather", "nice", "today"]), \
            f"Local round-trip failed: got '{transcript}'"


class TestProviderRegistryIntegration:
    """Test VoiceProviderRegistry with real providers."""

    @pytest.mark.asyncio
    async def test_registry_discovers_available_providers(self):
        """Registry should find whatever providers are actually installed."""
        from kestrel_sovereign.voice.provider_registry import VoiceProviderRegistry

        config = {
            "tts_provider_priority": ["piper", "openai", "elevenlabs"],
            "stt_provider_priority": ["faster_whisper", "openai", "deepgram"],
            "piper": {"data_dir": os.environ.get("PIPER_DATA_DIR", "/Volumes/data2/models/piper")},
            "faster_whisper": {
                "model": "tiny",
                "data_dir": os.environ.get("WHISPER_DATA_DIR", "/Volumes/data2/models/whisper"),
            },
        }
        registry = VoiceProviderRegistry(config=config)
        await registry.initialize()

        tts_names = registry.list_tts_providers()
        stt_names = registry.list_stt_providers()

        # At least log what we found
        print(f"Available TTS providers: {tts_names}")
        print(f"Available STT providers: {stt_names}")

        # If OpenAI key is set, OpenAI should be registered
        if os.environ.get("OPENAI_API_KEY"):
            assert "openai" in tts_names
            assert "openai" in stt_names

    @pytest.mark.asyncio
    async def test_local_providers_flagged_correctly(self):
        """Local providers must have is_local=True for privacy gate."""
        from kestrel_sovereign.voice.provider_registry import VoiceProviderRegistry

        config = {
            "tts_provider_priority": ["piper", "openai"],
            "stt_provider_priority": ["faster_whisper", "openai"],
            "piper": {"data_dir": os.environ.get("PIPER_DATA_DIR", "/Volumes/data2/models/piper")},
            "faster_whisper": {
                "model": "tiny",
                "data_dir": os.environ.get("WHISPER_DATA_DIR", "/Volumes/data2/models/whisper"),
            },
        }
        registry = VoiceProviderRegistry(config=config)
        await registry.initialize()

        local_tts = registry.get_local_tts()
        local_stt = registry.get_local_stt()

        for p in local_tts:
            assert p.is_local is True, f"{p.name} should be local"
        for p in local_stt:
            assert p.is_local is True, f"{p.name} should be local"

        # Cloud providers should NOT be in local lists
        for p in local_tts:
            assert p.name not in ("openai", "elevenlabs")
        for p in local_stt:
            assert p.name not in ("openai", "deepgram")
