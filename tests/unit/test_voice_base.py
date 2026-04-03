"""
Unit tests for voice provider abstraction layer.

Tests VoiceConfig serialization, VoiceProviderRegistry register/get/list,
local provider filtering, and missing provider behavior.
"""
import pytest
from typing import AsyncIterator

from kestrel_sovereign.voice import (
    TTSProvider,
    STTProvider,
    VoiceConfig,
    VoiceInfo,
    VoiceProviderRegistry,
)


# ---------------------------------------------------------------------------
# Concrete test implementations of the ABCs
# ---------------------------------------------------------------------------

class FakeTTSLocal(TTSProvider):
    """Fake local TTS provider for testing."""
    name = "fake_local_tts"
    is_local = True

    async def synthesize(self, text, voice_id, model="", output_format="opus"):
        return b"fake_audio"

    async def synthesize_stream(self, text, voice_id, model="", output_format="opus"):
        yield b"chunk1"
        yield b"chunk2"

    async def list_voices(self):
        return [VoiceInfo(voice_id="v1", name="Voice 1", provider=self.name)]

    async def is_available(self):
        return True


class FakeTTSCloud(TTSProvider):
    """Fake cloud TTS provider for testing."""
    name = "fake_cloud_tts"
    is_local = False

    async def synthesize(self, text, voice_id, model="", output_format="opus"):
        return b"cloud_audio"

    async def synthesize_stream(self, text, voice_id, model="", output_format="opus"):
        yield b"cloud_chunk"

    async def list_voices(self):
        return [VoiceInfo(voice_id="nova", name="Nova", provider=self.name)]

    async def is_available(self):
        return True


class FakeSTTLocal(STTProvider):
    """Fake local STT provider for testing."""
    name = "fake_local_stt"
    is_local = True

    async def transcribe(self, audio, language="", audio_format="opus"):
        return "transcribed text"

    async def transcribe_stream(self, audio_stream, language=""):
        yield "partial"
        yield " text"

    async def is_available(self):
        return True


class FakeSTTCloud(STTProvider):
    """Fake cloud STT provider for testing."""
    name = "fake_cloud_stt"
    is_local = False

    async def transcribe(self, audio, language="", audio_format="opus"):
        return "cloud transcription"

    async def transcribe_stream(self, audio_stream, language=""):
        yield "cloud partial"

    async def is_available(self):
        return True


# ---------------------------------------------------------------------------
# VoiceConfig tests
# ---------------------------------------------------------------------------

class TestVoiceConfig:
    def test_defaults(self):
        cfg = VoiceConfig()
        assert cfg.tts_provider == ""
        assert cfg.sample_rate == 24000
        assert cfg.output_format == "opus"

    def test_to_dict(self):
        cfg = VoiceConfig(tts_provider="piper", tts_voice_id="en_US-lessac-medium")
        d = cfg.to_dict()
        assert d["tts_provider"] == "piper"
        assert d["tts_voice_id"] == "en_US-lessac-medium"
        assert d["sample_rate"] == 24000

    def test_from_dict(self):
        data = {
            "tts_provider": "openai",
            "tts_voice_id": "nova",
            "tts_model": "tts-1-hd",
            "stt_provider": "faster_whisper",
            "stt_model": "large-v3",
            "sample_rate": 16000,
            "output_format": "mp3",
        }
        cfg = VoiceConfig.from_dict(data)
        assert cfg.tts_provider == "openai"
        assert cfg.tts_voice_id == "nova"
        assert cfg.sample_rate == 16000
        assert cfg.output_format == "mp3"

    def test_round_trip(self):
        original = VoiceConfig(
            tts_provider="piper",
            tts_voice_id="en_US-lessac-medium",
            tts_model="",
            stt_provider="faster_whisper",
            stt_model="large-v3",
            sample_rate=16000,
            output_format="wav",
        )
        restored = VoiceConfig.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_ignores_unknown_keys(self):
        data = {"tts_provider": "piper", "unknown_field": "should_be_ignored"}
        cfg = VoiceConfig.from_dict(data)
        assert cfg.tts_provider == "piper"
        assert not hasattr(cfg, "unknown_field")

    def test_from_dict_empty(self):
        cfg = VoiceConfig.from_dict({})
        assert cfg == VoiceConfig()


# ---------------------------------------------------------------------------
# VoiceInfo tests
# ---------------------------------------------------------------------------

class TestVoiceInfo:
    def test_creation(self):
        info = VoiceInfo(
            voice_id="nova",
            name="Nova",
            provider="openai",
            language="en",
            gender="feminine",
        )
        assert info.voice_id == "nova"
        assert info.gender == "feminine"

    def test_defaults(self):
        info = VoiceInfo(voice_id="v1", name="V1", provider="test")
        assert info.language == "en"
        assert info.gender == "neutral"
        assert info.preview_url == ""


# ---------------------------------------------------------------------------
# VoiceProviderRegistry tests
# ---------------------------------------------------------------------------

class TestVoiceProviderRegistry:
    def test_register_and_get_tts(self):
        registry = VoiceProviderRegistry(config={})
        provider = FakeTTSLocal()
        registry.register_tts(provider)
        assert registry.get_tts("fake_local_tts") is provider

    def test_register_and_get_stt(self):
        registry = VoiceProviderRegistry(config={})
        provider = FakeSTTLocal()
        registry.register_stt(provider)
        assert registry.get_stt("fake_local_stt") is provider

    def test_get_missing_tts_returns_none(self):
        registry = VoiceProviderRegistry(config={})
        assert registry.get_tts("nonexistent") is None

    def test_get_missing_stt_returns_none(self):
        registry = VoiceProviderRegistry(config={})
        assert registry.get_stt("nonexistent") is None

    def test_list_tts_providers(self):
        registry = VoiceProviderRegistry(config={})
        registry.register_tts(FakeTTSLocal())
        registry.register_tts(FakeTTSCloud())
        names = registry.list_tts_providers()
        assert "fake_local_tts" in names
        assert "fake_cloud_tts" in names
        assert len(names) == 2

    def test_list_stt_providers(self):
        registry = VoiceProviderRegistry(config={})
        registry.register_stt(FakeSTTLocal())
        registry.register_stt(FakeSTTCloud())
        names = registry.list_stt_providers()
        assert "fake_local_stt" in names
        assert "fake_cloud_stt" in names
        assert len(names) == 2

    def test_list_empty(self):
        registry = VoiceProviderRegistry(config={})
        assert registry.list_tts_providers() == []
        assert registry.list_stt_providers() == []

    def test_get_local_tts(self):
        registry = VoiceProviderRegistry(config={})
        local = FakeTTSLocal()
        cloud = FakeTTSCloud()
        registry.register_tts(local)
        registry.register_tts(cloud)
        local_providers = registry.get_local_tts()
        assert len(local_providers) == 1
        assert local_providers[0] is local

    def test_get_local_stt(self):
        registry = VoiceProviderRegistry(config={})
        local = FakeSTTLocal()
        cloud = FakeSTTCloud()
        registry.register_stt(local)
        registry.register_stt(cloud)
        local_providers = registry.get_local_stt()
        assert len(local_providers) == 1
        assert local_providers[0] is local

    def test_get_local_tts_empty_when_no_local(self):
        registry = VoiceProviderRegistry(config={})
        registry.register_tts(FakeTTSCloud())
        assert registry.get_local_tts() == []

    def test_get_local_stt_empty_when_no_local(self):
        registry = VoiceProviderRegistry(config={})
        registry.register_stt(FakeSTTCloud())
        assert registry.get_local_stt() == []

    def test_register_overwrites_same_name(self):
        registry = VoiceProviderRegistry(config={})
        p1 = FakeTTSLocal()
        p2 = FakeTTSLocal()
        registry.register_tts(p1)
        registry.register_tts(p2)
        assert registry.get_tts("fake_local_tts") is p2
        assert len(registry.list_tts_providers()) == 1


# ---------------------------------------------------------------------------
# ABC enforcement tests
# ---------------------------------------------------------------------------

class TestABCEnforcement:
    def test_cannot_instantiate_tts_provider(self):
        with pytest.raises(TypeError):
            TTSProvider()

    def test_cannot_instantiate_stt_provider(self):
        with pytest.raises(TypeError):
            STTProvider()


# ---------------------------------------------------------------------------
# Async provider method tests
# ---------------------------------------------------------------------------

class TestProviderMethods:
    @pytest.mark.asyncio
    async def test_tts_synthesize(self):
        provider = FakeTTSLocal()
        result = await provider.synthesize("hello", "v1")
        assert result == b"fake_audio"

    @pytest.mark.asyncio
    async def test_tts_synthesize_stream(self):
        provider = FakeTTSLocal()
        chunks = []
        async for chunk in provider.synthesize_stream("hello", "v1"):
            chunks.append(chunk)
        assert chunks == [b"chunk1", b"chunk2"]

    @pytest.mark.asyncio
    async def test_tts_list_voices(self):
        provider = FakeTTSLocal()
        voices = await provider.list_voices()
        assert len(voices) == 1
        assert voices[0].voice_id == "v1"

    @pytest.mark.asyncio
    async def test_tts_is_available(self):
        provider = FakeTTSLocal()
        assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_stt_transcribe(self):
        provider = FakeSTTLocal()
        result = await provider.transcribe(b"audio_data")
        assert result == "transcribed text"

    @pytest.mark.asyncio
    async def test_stt_transcribe_stream(self):
        provider = FakeSTTLocal()

        async def audio_gen():
            yield b"chunk"

        segments = []
        async for segment in provider.transcribe_stream(audio_gen()):
            segments.append(segment)
        assert segments == ["partial", " text"]

    @pytest.mark.asyncio
    async def test_stt_is_available(self):
        provider = FakeSTTLocal()
        assert await provider.is_available() is True


# ---------------------------------------------------------------------------
# Registry initialize() test
# ---------------------------------------------------------------------------

class TestRegistryInitialize:
    @pytest.mark.asyncio
    async def test_initialize_skips_unknown_providers(self, monkeypatch):
        """Initialize with unknown providers should log warnings but not fail."""
        # Mock entry_point discovery so installed packages don't interfere
        monkeypatch.setattr(
            "kestrel_sovereign.voice.provider_registry.discover_entry_point_classes",
            lambda *args, **kwargs: {},
        )
        config = {
            "tts_provider_priority": ["nonexistent_tts"],
            "stt_provider_priority": ["nonexistent_stt"],
        }
        registry = VoiceProviderRegistry(config=config)
        await registry.initialize()
        assert registry.list_tts_providers() == []
        assert registry.list_stt_providers() == []

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, monkeypatch):
        """Calling initialize() twice should not re-initialize."""
        monkeypatch.setattr(
            "kestrel_sovereign.voice.provider_registry.discover_entry_point_classes",
            lambda *args, **kwargs: {},
        )
        config = {"tts_provider_priority": [], "stt_provider_priority": []}
        registry = VoiceProviderRegistry(config=config)
        await registry.initialize()
        # Manually register after init
        registry.register_tts(FakeTTSLocal())
        await registry.initialize()  # Should be a no-op
        assert len(registry.list_tts_providers()) == 1
