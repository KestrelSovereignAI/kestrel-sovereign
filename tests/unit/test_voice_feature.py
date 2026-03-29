"""
Unit tests for VoiceFeature.

Tests:
- Feature discovery (auto-discovered by feature scan)
- Tool registration (4 tools: list_voices, set_voice, speak, transcribe)
- list_voices with mock registry
- set_voice persistence
- speak with mock TTS provider
- transcribe with mock STT provider
- Privacy gate (cloud blocked in ephemeral mode)
"""

import pytest
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.features.voice.feature import VoiceFeature, VoicePrivacyError
from kestrel_sovereign.voice.base import (
    TTSProvider,
    STTProvider,
    VoiceConfig,
    VoiceInfo,
)
from kestrel_sovereign.voice.provider_registry import VoiceProviderRegistry


# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------

class FakeLocalTTS(TTSProvider):
    name = "fake_local"
    is_local = True

    async def synthesize(self, text, voice_id, model="", output_format="opus"):
        return b"local_audio_bytes"

    async def synthesize_stream(self, text, voice_id, model="", output_format="opus"):
        yield b"chunk"

    async def list_voices(self):
        return [
            VoiceInfo(voice_id="local-v1", name="Local Voice 1", provider=self.name),
            VoiceInfo(voice_id="local-v2", name="Local Voice 2", provider=self.name),
        ]

    async def is_available(self):
        return True


class FakeCloudTTS(TTSProvider):
    name = "fake_cloud"
    is_local = False

    async def synthesize(self, text, voice_id, model="", output_format="opus"):
        return b"cloud_audio_bytes"

    async def synthesize_stream(self, text, voice_id, model="", output_format="opus"):
        yield b"cloud_chunk"

    async def list_voices(self):
        return [
            VoiceInfo(voice_id="nova", name="Nova", provider=self.name),
        ]

    async def is_available(self):
        return True


class FakeLocalSTT(STTProvider):
    name = "fake_local_stt"
    is_local = True

    async def transcribe(self, audio, language="", audio_format="opus"):
        return "transcribed text from local"

    async def transcribe_stream(self, audio_stream, language=""):
        yield "partial"

    async def is_available(self):
        return True


class FakeCloudSTT(STTProvider):
    name = "fake_cloud_stt"
    is_local = False

    async def transcribe(self, audio, language="", audio_format="opus"):
        return "transcribed text from cloud"

    async def transcribe_stream(self, audio_stream, language=""):
        yield "partial"

    async def is_available(self):
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_agent(cloud_allowed=True, has_storage=True, has_identity=False, voice_config_dict=None):
    """Build a mock agent with configurable privacy and storage."""
    agent = MagicMock()
    agent.agent_id = "test-voice-agent"
    agent.config = {"voice": {}}

    # Privacy
    privacy_agent = MagicMock()
    privacy_config = MagicMock()
    privacy_config.allows_cloud_llm.return_value = cloud_allowed
    privacy_agent.privacy_config = privacy_config
    agent.privacy_agent = privacy_agent

    # Storage
    if has_storage:
        storage = AsyncMock()
        storage.store_file = AsyncMock(return_value="sha256_abc123")
        storage.retrieve_file = AsyncMock(return_value=b"stored_audio")
        storage.get_file_metadata = AsyncMock(return_value={"format": "opus"})
        agent.storage = storage
    else:
        agent.storage = None

    # Identity
    if has_identity:
        identity = MagicMock()
        identity.voice_config = voice_config_dict or {}
        agent.identity = identity
    else:
        agent.identity = None

    return agent


def _make_registry(local_tts=True, cloud_tts=True, local_stt=True, cloud_stt=True):
    """Build a VoiceProviderRegistry with fake providers pre-registered."""
    registry = VoiceProviderRegistry({})
    registry._initialized = True
    if local_tts:
        registry.register_tts(FakeLocalTTS())
    if cloud_tts:
        registry.register_tts(FakeCloudTTS())
    if local_stt:
        registry.register_stt(FakeLocalSTT())
    if cloud_stt:
        registry.register_stt(FakeCloudSTT())
    return registry


@pytest.fixture
def agent():
    return _make_agent()


@pytest.fixture
def feature(agent):
    f = VoiceFeature(agent)
    # Pre-set internals so tests don't hit real registry init
    f._voice_registry = _make_registry()
    f._voice_config = VoiceConfig()
    return f


# ---------------------------------------------------------------------------
# Feature discovery
# ---------------------------------------------------------------------------

class TestFeatureDiscovery:
    def test_voice_feature_is_discovered(self):
        """VoiceFeature should be found by the feature discovery scan."""
        from kestrel_sovereign.features import discover_feature_modules
        modules = discover_feature_modules()
        assert "kestrel_sovereign.features.voice.feature" in modules

    def test_voice_feature_class_found(self):
        """find_feature_class should return VoiceFeature from the voice module."""
        import importlib
        from kestrel_sovereign.features import find_feature_class
        mod = importlib.import_module("kestrel_sovereign.features.voice.feature")
        cls = find_feature_class(mod)
        assert cls is VoiceFeature


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_has_four_tools(self, feature):
        tools = feature.get_tools()
        assert len(tools) == 4

    def test_tool_names(self, feature):
        names = {t.name for t in feature.get_tools()}
        assert names == {"list_voices", "set_voice", "speak", "transcribe"}

    def test_tool_description_property(self, feature):
        assert "text-to-speech" in feature.tool_description.lower()

    def test_tool_name_is_snake_case(self, feature):
        assert feature.tool_name == "voice_feature"


# ---------------------------------------------------------------------------
# list_voices
# ---------------------------------------------------------------------------

class TestListVoices:
    @pytest.mark.asyncio
    async def test_list_all_voices(self, feature):
        result = await feature.list_voices()
        assert result["count"] == 3  # 2 local + 1 cloud
        ids = [v["voice_id"] for v in result["voices"]]
        assert "local-v1" in ids
        assert "nova" in ids

    @pytest.mark.asyncio
    async def test_filter_by_provider(self, feature):
        result = await feature.list_voices(provider="fake_local")
        assert result["count"] == 2
        assert all(v["provider"] == "fake_local" for v in result["voices"])

    @pytest.mark.asyncio
    async def test_privacy_filters_cloud(self, feature):
        """Cloud voices hidden when privacy blocks cloud."""
        feature.agent.privacy_agent.privacy_config.allows_cloud_llm.return_value = False
        result = await feature.list_voices()
        providers = {v["provider"] for v in result["voices"]}
        assert "fake_cloud" not in providers
        assert result["count"] == 2  # only local voices


# ---------------------------------------------------------------------------
# set_voice
# ---------------------------------------------------------------------------

class TestSetVoice:
    @pytest.mark.asyncio
    async def test_set_known_voice(self, feature):
        result = await feature.set_voice(voice_id="nova")
        assert result["success"] is True
        assert result["voice_id"] == "nova"
        assert result["provider"] == "fake_cloud"
        assert feature._voice_config.tts_voice_id == "nova"
        assert feature._voice_config.tts_provider == "fake_cloud"

    @pytest.mark.asyncio
    async def test_set_voice_explicit_provider(self, feature):
        result = await feature.set_voice(voice_id="local-v1", provider="fake_local")
        assert result["success"] is True
        assert result["provider"] == "fake_local"

    @pytest.mark.asyncio
    async def test_set_unknown_voice(self, feature):
        result = await feature.set_voice(voice_id="nonexistent")
        assert result["success"] is False
        assert "not find" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_set_cloud_voice_blocked_by_privacy(self, feature):
        feature.agent.privacy_agent.privacy_config.allows_cloud_llm.return_value = False
        result = await feature.set_voice(voice_id="nova", provider="fake_cloud")
        assert result["success"] is False
        assert "privacy" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_set_voice_persists_to_identity(self):
        agent = _make_agent(has_identity=True, voice_config_dict={})
        f = VoiceFeature(agent)
        f._voice_registry = _make_registry()
        f._voice_config = VoiceConfig()
        result = await f.set_voice(voice_id="nova")
        assert result["success"] is True
        assert agent.identity.voice_config["tts_voice_id"] == "nova"


# ---------------------------------------------------------------------------
# speak
# ---------------------------------------------------------------------------

class TestSpeak:
    @pytest.mark.asyncio
    async def test_speak_returns_metadata(self, feature):
        feature._voice_config.tts_voice_id = "local-v1"
        feature._voice_config.tts_provider = "fake_local"
        result = await feature.speak(text="Hello world")
        assert result["success"] is True
        assert result["content_hash"] == "sha256_abc123"
        assert result["voice_id"] == "local-v1"
        assert result["provider"] == "fake_local"
        assert result["text_length"] == 11
        assert result["audio_size"] == len(b"local_audio_bytes")

    @pytest.mark.asyncio
    async def test_speak_auto_selects_voice(self, feature):
        """When no voice_id is set, pick the first available."""
        feature._voice_config.tts_provider = "fake_local"
        feature._voice_config.tts_voice_id = ""
        result = await feature.speak(text="Auto voice")
        assert result["success"] is True
        assert result["voice_id"] == "local-v1"

    @pytest.mark.asyncio
    async def test_speak_stores_via_storage(self, feature):
        feature._voice_config.tts_voice_id = "local-v1"
        feature._voice_config.tts_provider = "fake_local"
        await feature.speak(text="Store me")
        feature.agent.storage.store_file.assert_awaited_once()
        call_args = feature.agent.storage.store_file.call_args
        assert call_args[0][0] == b"local_audio_bytes"
        assert "speech" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_speak_no_storage_warns(self):
        agent = _make_agent(has_storage=False)
        f = VoiceFeature(agent)
        f._voice_registry = _make_registry()
        f._voice_config = VoiceConfig(tts_provider="fake_local", tts_voice_id="local-v1")
        result = await f.speak(text="No storage")
        assert result["success"] is True
        assert result["content_hash"] == ""

    @pytest.mark.asyncio
    async def test_speak_blocked_by_privacy(self):
        """Speak fails when only cloud TTS is available and privacy blocks cloud."""
        agent = _make_agent(cloud_allowed=False)
        f = VoiceFeature(agent)
        f._voice_registry = _make_registry(local_tts=False, cloud_tts=True)
        f._voice_config = VoiceConfig()
        with pytest.raises(VoicePrivacyError, match="local TTS"):
            await f.speak(text="blocked")


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------

class TestTranscribe:
    @pytest.mark.asyncio
    async def test_transcribe_success(self, feature):
        feature._voice_config.stt_provider = "fake_local_stt"
        result = await feature.transcribe(audio_content_hash="hash123")
        assert result["success"] is True
        assert result["text"] == "transcribed text from local"
        assert result["provider"] == "fake_local_stt"

    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self, feature):
        feature.agent.storage.retrieve_file = AsyncMock(return_value=None)
        result = await feature.transcribe(audio_content_hash="missing")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_transcribe_no_storage(self):
        agent = _make_agent(has_storage=False)
        f = VoiceFeature(agent)
        f._voice_registry = _make_registry()
        f._voice_config = VoiceConfig()
        result = await f.transcribe(audio_content_hash="hash123")
        assert result["success"] is False
        assert "storage" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_transcribe_blocked_by_privacy(self):
        """Transcribe fails when only cloud STT available and privacy blocks cloud."""
        agent = _make_agent(cloud_allowed=False)
        f = VoiceFeature(agent)
        f._voice_registry = _make_registry(local_stt=False, cloud_stt=True)
        f._voice_config = VoiceConfig()
        with pytest.raises(VoicePrivacyError, match="local STT"):
            await f.transcribe(audio_content_hash="hash123")


# ---------------------------------------------------------------------------
# Privacy gate
# ---------------------------------------------------------------------------

class TestPrivacyGate:
    @pytest.mark.asyncio
    async def test_cloud_allowed_returns_cloud_tts(self):
        agent = _make_agent(cloud_allowed=True)
        f = VoiceFeature(agent)
        f._voice_registry = _make_registry(local_tts=False, cloud_tts=True)
        f._voice_config = VoiceConfig()
        provider = await f._get_tts_provider()
        assert provider.name == "fake_cloud"
        assert not provider.is_local

    @pytest.mark.asyncio
    async def test_cloud_blocked_returns_local_tts(self):
        agent = _make_agent(cloud_allowed=False)
        f = VoiceFeature(agent)
        f._voice_registry = _make_registry(local_tts=True, cloud_tts=True)
        f._voice_config = VoiceConfig()
        provider = await f._get_tts_provider()
        assert provider.is_local

    @pytest.mark.asyncio
    async def test_cloud_blocked_no_local_raises(self):
        agent = _make_agent(cloud_allowed=False)
        f = VoiceFeature(agent)
        f._voice_registry = _make_registry(local_tts=False, cloud_tts=True, local_stt=False, cloud_stt=True)
        f._voice_config = VoiceConfig()
        with pytest.raises(VoicePrivacyError):
            await f._get_tts_provider()
        with pytest.raises(VoicePrivacyError):
            await f._get_stt_provider()

    @pytest.mark.asyncio
    async def test_no_privacy_agent_allows_cloud(self):
        agent = _make_agent(cloud_allowed=True)
        agent.privacy_agent = None
        f = VoiceFeature(agent)
        f._voice_registry = _make_registry(local_tts=False, cloud_tts=True)
        f._voice_config = VoiceConfig()
        provider = await f._get_tts_provider()
        assert provider.name == "fake_cloud"


# ---------------------------------------------------------------------------
# Initialize with identity
# ---------------------------------------------------------------------------

class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_loads_voice_config_from_identity(self):
        agent = _make_agent(has_identity=True, voice_config_dict={
            "tts_provider": "openai",
            "tts_voice_id": "nova",
            "output_format": "mp3",
        })
        f = VoiceFeature(agent)
        await f.initialize()
        assert f._voice_config.tts_provider == "openai"
        assert f._voice_config.tts_voice_id == "nova"
        assert f._voice_config.output_format == "mp3"

    @pytest.mark.asyncio
    async def test_initialize_no_identity(self):
        agent = _make_agent(has_identity=False)
        f = VoiceFeature(agent)
        await f.initialize()
        assert f._voice_config.tts_provider == ""
        assert f._voice_config.tts_voice_id == ""
