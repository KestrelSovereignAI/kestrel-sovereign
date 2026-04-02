"""
Unit tests for voice privacy gate — auto-switch providers by privacy mode.

Tests:
- Each privacy preset with cloud/local provider combinations
- Auto-fallback behavior when privacy mode changes
- Clear, actionable error messages
- Audio storage policy enforcement per privacy mode
- 403 responses for blocked voice operations (endpoint-level)
- Biometric warning when enabling cloud voice providers
- Provider validation on config POST
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.features.voice.feature import VoiceFeature, VoicePrivacyError
from kestrel_sovereign.privacy import PrivacyConfig, PRIVACY_PRESETS, get_privacy_preset
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
    name = "piper"
    is_local = True

    async def synthesize(self, text, voice_id, model="", output_format="opus"):
        return b"local_audio"

    async def synthesize_stream(self, text, voice_id, model="", output_format="opus"):
        yield b"chunk"

    async def list_voices(self):
        return [VoiceInfo(voice_id="en_US-lessac-medium", name="Lessac", provider=self.name)]

    async def is_available(self):
        return True


class FakeCloudTTS(TTSProvider):
    name = "openai"
    is_local = False

    async def synthesize(self, text, voice_id, model="", output_format="opus"):
        return b"cloud_audio"

    async def synthesize_stream(self, text, voice_id, model="", output_format="opus"):
        yield b"cloud_chunk"

    async def list_voices(self):
        return [VoiceInfo(voice_id="nova", name="Nova", provider=self.name)]

    async def is_available(self):
        return True


class FakeLocalSTT(STTProvider):
    name = "faster_whisper"
    is_local = True

    async def transcribe(self, audio, language="", audio_format="opus"):
        return "local transcription"

    async def transcribe_stream(self, audio_stream, language=""):
        yield "partial"

    async def is_available(self):
        return True


class FakeCloudSTT(STTProvider):
    name = "deepgram"
    is_local = False

    async def transcribe(self, audio, language="", audio_format="opus"):
        return "cloud transcription"

    async def transcribe_stream(self, audio_stream, language=""):
        yield "partial"

    async def is_available(self):
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(privacy_preset: str = "normal", has_storage: bool = True):
    """Build a mock agent with a specific privacy preset."""
    agent = MagicMock()
    agent.agent_id = "test-voice-agent"
    agent.config = {"voice": {}}

    privacy_agent = MagicMock()
    config = get_privacy_preset(privacy_preset)
    privacy_agent.privacy_config = config
    # Wire up the unified PrivacyAgent API methods that VoiceFeature delegates to
    privacy_agent.can_use_cloud.return_value = config.allows_cloud_llm()
    privacy_agent.get_mode_name.return_value = privacy_preset
    privacy_agent.get_storage_policy.return_value = config.storage
    privacy_agent.can_store.side_effect = lambda data_type="conversation": (
        not config.is_ephemeral() and not config.uses_temp_storage()
        if data_type != "metadata" else True
    )
    privacy_agent.requires_anonymization.return_value = config.requires_anonymization()
    agent.privacy_agent = privacy_agent

    if has_storage:
        storage = AsyncMock()
        storage.store_file = AsyncMock(return_value="sha256_abc123")
        storage.retrieve_file = AsyncMock(return_value=b"stored_audio")
        storage.get_file_metadata = AsyncMock(return_value={"format": "opus"})
        agent.storage = storage
    else:
        agent.storage = None

    agent.identity = None
    return agent


def _make_registry(local_tts=True, cloud_tts=True, local_stt=True, cloud_stt=True):
    """Build a VoiceProviderRegistry with fake providers."""
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


def _make_feature(privacy_preset="normal", local_tts=True, cloud_tts=True,
                  local_stt=True, cloud_stt=True, has_storage=True,
                  tts_provider="", stt_provider=""):
    """Build a VoiceFeature with configurable privacy and providers."""
    agent = _make_agent(privacy_preset, has_storage=has_storage)
    f = VoiceFeature(agent)
    f._voice_registry = _make_registry(local_tts, cloud_tts, local_stt, cloud_stt)
    f._voice_config = VoiceConfig(tts_provider=tts_provider, stt_provider=stt_provider)
    return f


# ---------------------------------------------------------------------------
# Privacy preset enforcement — TTS
# ---------------------------------------------------------------------------

class TestPrivacyPresetTTS:
    """Test TTS provider selection across all 5 privacy presets."""

    @pytest.mark.asyncio
    async def test_ephemeral_blocks_cloud_tts(self):
        f = _make_feature("ephemeral", local_tts=True, cloud_tts=True, tts_provider="openai")
        with pytest.raises(VoicePrivacyError, match="Cannot use Openai TTS in ephemeral"):
            await f._get_tts_provider()

    @pytest.mark.asyncio
    async def test_ephemeral_allows_local_tts(self):
        f = _make_feature("ephemeral", local_tts=True, cloud_tts=True)
        provider = await f._get_tts_provider()
        assert provider.is_local
        assert provider.name == "piper"

    @pytest.mark.asyncio
    async def test_isolated_blocks_cloud_tts(self):
        f = _make_feature("isolated", local_tts=True, cloud_tts=True, tts_provider="openai")
        with pytest.raises(VoicePrivacyError, match="Cannot use Openai TTS in isolated"):
            await f._get_tts_provider()

    @pytest.mark.asyncio
    async def test_isolated_allows_local_tts(self):
        f = _make_feature("isolated", local_tts=True, cloud_tts=True)
        provider = await f._get_tts_provider()
        assert provider.is_local

    @pytest.mark.asyncio
    async def test_anonymous_allows_cloud_tts(self):
        f = _make_feature("anonymous", local_tts=True, cloud_tts=True, tts_provider="openai")
        provider = await f._get_tts_provider()
        assert provider.name == "openai"

    @pytest.mark.asyncio
    async def test_normal_allows_cloud_tts(self):
        f = _make_feature("normal", local_tts=True, cloud_tts=True, tts_provider="openai")
        provider = await f._get_tts_provider()
        assert provider.name == "openai"

    @pytest.mark.asyncio
    async def test_public_allows_cloud_tts(self):
        f = _make_feature("public", local_tts=True, cloud_tts=True, tts_provider="openai")
        provider = await f._get_tts_provider()
        assert provider.name == "openai"


# ---------------------------------------------------------------------------
# Privacy preset enforcement — STT
# ---------------------------------------------------------------------------

class TestPrivacyPresetSTT:
    """Test STT provider selection across all 5 privacy presets."""

    @pytest.mark.asyncio
    async def test_ephemeral_blocks_cloud_stt(self):
        f = _make_feature("ephemeral", local_stt=True, cloud_stt=True, stt_provider="deepgram")
        with pytest.raises(VoicePrivacyError, match="Cannot use Deepgram STT in ephemeral"):
            await f._get_stt_provider()

    @pytest.mark.asyncio
    async def test_ephemeral_allows_local_stt(self):
        f = _make_feature("ephemeral", local_stt=True, cloud_stt=True)
        provider = await f._get_stt_provider()
        assert provider.is_local
        assert provider.name == "faster_whisper"

    @pytest.mark.asyncio
    async def test_isolated_blocks_cloud_stt(self):
        f = _make_feature("isolated", local_stt=True, cloud_stt=True, stt_provider="deepgram")
        with pytest.raises(VoicePrivacyError, match="Cannot use Deepgram STT in isolated"):
            await f._get_stt_provider()

    @pytest.mark.asyncio
    async def test_anonymous_allows_cloud_stt(self):
        f = _make_feature("anonymous", local_stt=True, cloud_stt=True, stt_provider="deepgram")
        provider = await f._get_stt_provider()
        assert provider.name == "deepgram"

    @pytest.mark.asyncio
    async def test_normal_allows_cloud_stt(self):
        f = _make_feature("normal", local_stt=True, cloud_stt=True, stt_provider="deepgram")
        provider = await f._get_stt_provider()
        assert provider.name == "deepgram"

    @pytest.mark.asyncio
    async def test_public_allows_cloud_stt(self):
        f = _make_feature("public", local_stt=True, cloud_stt=True, stt_provider="deepgram")
        provider = await f._get_stt_provider()
        assert provider.name == "deepgram"


# ---------------------------------------------------------------------------
# Auto-fallback behavior
# ---------------------------------------------------------------------------

class TestAutoFallback:
    """Test on_privacy_mode_changed auto-switch behavior."""

    @pytest.mark.asyncio
    async def test_auto_switch_tts_to_local(self):
        f = _make_feature("ephemeral", local_tts=True, cloud_tts=True,
                          tts_provider="openai")
        result = await f.on_privacy_mode_changed()
        assert result is not None
        assert result["tts_switched"]["from"] == "openai"
        assert result["tts_switched"]["to"] == "piper"
        assert f._voice_config.tts_provider == "piper"

    @pytest.mark.asyncio
    async def test_auto_switch_stt_to_local(self):
        f = _make_feature("isolated", local_stt=True, cloud_stt=True,
                          stt_provider="deepgram")
        result = await f.on_privacy_mode_changed()
        assert result is not None
        assert result["stt_switched"]["from"] == "deepgram"
        assert result["stt_switched"]["to"] == "faster_whisper"
        assert f._voice_config.stt_provider == "faster_whisper"

    @pytest.mark.asyncio
    async def test_auto_switch_both_providers(self):
        f = _make_feature("ephemeral", local_tts=True, cloud_tts=True,
                          local_stt=True, cloud_stt=True,
                          tts_provider="openai", stt_provider="deepgram")
        result = await f.on_privacy_mode_changed()
        assert "tts_switched" in result
        assert "stt_switched" in result

    @pytest.mark.asyncio
    async def test_auto_switch_no_local_disables_tts(self):
        f = _make_feature("ephemeral", local_tts=False, cloud_tts=True,
                          tts_provider="openai")
        result = await f.on_privacy_mode_changed()
        assert result["tts_switched"]["from"] == "openai"
        assert result["tts_switched"]["to"] is None
        assert f._voice_config.tts_provider == ""

    @pytest.mark.asyncio
    async def test_auto_switch_no_local_disables_stt(self):
        f = _make_feature("ephemeral", local_stt=False, cloud_stt=True,
                          stt_provider="deepgram")
        result = await f.on_privacy_mode_changed()
        assert result["stt_switched"]["from"] == "deepgram"
        assert result["stt_switched"]["to"] is None
        assert f._voice_config.stt_provider == ""

    @pytest.mark.asyncio
    async def test_no_switch_needed_when_already_local(self):
        f = _make_feature("ephemeral", local_tts=True, cloud_tts=True,
                          tts_provider="piper")
        result = await f.on_privacy_mode_changed()
        assert result is None
        assert f._voice_config.tts_provider == "piper"

    @pytest.mark.asyncio
    async def test_no_switch_needed_when_cloud_allowed(self):
        f = _make_feature("normal", local_tts=True, cloud_tts=True,
                          tts_provider="openai")
        result = await f.on_privacy_mode_changed()
        assert result is None

    @pytest.mark.asyncio
    async def test_no_switch_when_no_provider_configured(self):
        f = _make_feature("ephemeral", local_tts=True, cloud_tts=True)
        result = await f.on_privacy_mode_changed()
        assert result is None

    @pytest.mark.asyncio
    async def test_auto_switch_persists_to_identity(self):
        agent = _make_agent("ephemeral")
        identity = MagicMock()
        identity.voice_config = {}
        agent.identity = identity
        f = VoiceFeature(agent)
        f._voice_registry = _make_registry(local_tts=True, cloud_tts=True)
        f._voice_config = VoiceConfig(tts_provider="openai")
        await f.on_privacy_mode_changed()
        assert identity.voice_config["tts_provider"] == "piper"


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------

class TestErrorMessages:
    """Test that error messages are clear and actionable."""

    @pytest.mark.asyncio
    async def test_tts_error_mentions_provider_name(self):
        f = _make_feature("ephemeral", local_tts=True, cloud_tts=True, tts_provider="openai")
        with pytest.raises(VoicePrivacyError) as exc_info:
            await f._get_tts_provider()
        msg = str(exc_info.value)
        assert "Openai TTS" in msg
        assert "ephemeral" in msg
        assert "piper-tts" in msg
        assert "anonymous" in msg

    @pytest.mark.asyncio
    async def test_stt_error_mentions_provider_name(self):
        f = _make_feature("isolated", local_stt=True, cloud_stt=True, stt_provider="deepgram")
        with pytest.raises(VoicePrivacyError) as exc_info:
            await f._get_stt_provider()
        msg = str(exc_info.value)
        assert "Deepgram STT" in msg
        assert "isolated" in msg
        assert "faster-whisper" in msg
        assert "anonymous" in msg

    @pytest.mark.asyncio
    async def test_no_local_tts_error_message(self):
        f = _make_feature("ephemeral", local_tts=False, cloud_tts=True)
        with pytest.raises(VoicePrivacyError) as exc_info:
            await f._get_tts_provider()
        msg = str(exc_info.value)
        assert "No local TTS" in msg
        assert "piper-tts" in msg

    @pytest.mark.asyncio
    async def test_no_local_stt_error_message(self):
        f = _make_feature("ephemeral", local_stt=False, cloud_stt=True)
        with pytest.raises(VoicePrivacyError) as exc_info:
            await f._get_stt_provider()
        msg = str(exc_info.value)
        assert "No local STT" in msg
        assert "faster-whisper" in msg


# ---------------------------------------------------------------------------
# Audio storage policy
# ---------------------------------------------------------------------------

class TestAudioStoragePolicy:
    """Test audio storage respects privacy config."""

    def test_ephemeral_storage_policy(self):
        f = _make_feature("ephemeral")
        assert f._get_audio_storage_policy() == "none"

    def test_isolated_storage_policy(self):
        f = _make_feature("isolated")
        assert f._get_audio_storage_policy() == "temp"

    def test_anonymous_storage_policy(self):
        f = _make_feature("anonymous")
        assert f._get_audio_storage_policy() == "scrubbed"

    def test_normal_storage_policy(self):
        f = _make_feature("normal")
        assert f._get_audio_storage_policy() == "full"

    def test_public_storage_policy(self):
        f = _make_feature("public")
        assert f._get_audio_storage_policy() == "full"

    def test_no_privacy_agent_defaults_to_full(self):
        f = _make_feature("normal")
        f.agent.privacy_agent = None
        assert f._get_audio_storage_policy() == "full"

    @pytest.mark.asyncio
    async def test_ephemeral_speak_no_storage_call(self):
        """In ephemeral mode, speak should NOT call storage.store_file."""
        f = _make_feature("ephemeral", local_tts=True, cloud_tts=False)
        f._voice_config.tts_voice_id = "en_US-lessac-medium"
        f._voice_config.tts_provider = "piper"
        result = await f.speak(text="Hello")
        assert result["success"] is True
        assert result["content_hash"] == ""
        assert result["storage_policy"] == "none"
        f.agent.storage.store_file.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_isolated_speak_stores_with_temp_policy(self):
        """In isolated mode, speak stores with temp policy metadata."""
        f = _make_feature("isolated", local_tts=True, cloud_tts=False)
        f._voice_config.tts_voice_id = "en_US-lessac-medium"
        f._voice_config.tts_provider = "piper"
        result = await f.speak(text="Hello")
        assert result["success"] is True
        assert result["storage_policy"] == "temp"
        f.agent.storage.store_file.assert_awaited_once()
        call_kwargs = f.agent.storage.store_file.call_args
        metadata = call_kwargs[1].get("metadata") if call_kwargs[1] else call_kwargs[0][2]
        assert metadata["storage_policy"] == "temp"

    @pytest.mark.asyncio
    async def test_anonymous_speak_scrubs_metadata(self):
        """In anonymous mode, speak stores but omits voice_id/provider from metadata."""
        f = _make_feature("anonymous", local_tts=True, cloud_tts=True)
        f._voice_config.tts_voice_id = "en_US-lessac-medium"
        f._voice_config.tts_provider = "piper"
        result = await f.speak(text="Hello")
        assert result["success"] is True
        assert result["storage_policy"] == "scrubbed"
        f.agent.storage.store_file.assert_awaited_once()
        call_args = f.agent.storage.store_file.call_args
        metadata = call_args[1].get("metadata") if call_args[1] else call_args[0][2]
        assert "voice_id" not in metadata
        assert "provider" not in metadata
        assert metadata["storage_policy"] == "scrubbed"

    @pytest.mark.asyncio
    async def test_normal_speak_stores_full_metadata(self):
        """In normal mode, speak stores with full metadata."""
        f = _make_feature("normal", local_tts=True, cloud_tts=True)
        f._voice_config.tts_voice_id = "en_US-lessac-medium"
        f._voice_config.tts_provider = "piper"
        result = await f.speak(text="Hello")
        assert result["success"] is True
        assert result["storage_policy"] == "full"
        f.agent.storage.store_file.assert_awaited_once()
        call_args = f.agent.storage.store_file.call_args
        metadata = call_args[1].get("metadata") if call_args[1] else call_args[0][2]
        assert "voice_id" in metadata
        assert "provider" in metadata


# ---------------------------------------------------------------------------
# Provider allowed check
# ---------------------------------------------------------------------------

class TestIsProviderAllowed:
    """Test is_provider_allowed helper."""

    def test_cloud_tts_blocked_in_ephemeral(self):
        f = _make_feature("ephemeral", local_tts=True, cloud_tts=True)
        assert f.is_provider_allowed("openai", "tts") is False
        assert f.is_provider_allowed("piper", "tts") is True

    def test_cloud_stt_blocked_in_isolated(self):
        f = _make_feature("isolated", local_stt=True, cloud_stt=True)
        assert f.is_provider_allowed("deepgram", "stt") is False
        assert f.is_provider_allowed("faster_whisper", "stt") is True

    def test_all_allowed_in_anonymous(self):
        f = _make_feature("anonymous", local_tts=True, cloud_tts=True,
                          local_stt=True, cloud_stt=True)
        assert f.is_provider_allowed("openai", "tts") is True
        assert f.is_provider_allowed("piper", "tts") is True
        assert f.is_provider_allowed("deepgram", "stt") is True
        assert f.is_provider_allowed("faster_whisper", "stt") is True

    def test_unknown_provider_allowed(self):
        """Unknown providers (not in registry) are allowed by default."""
        f = _make_feature("ephemeral")
        assert f.is_provider_allowed("unknown_provider", "tts") is True


# ---------------------------------------------------------------------------
# Privacy mode name resolution
# ---------------------------------------------------------------------------

class TestPrivacyModeName:
    """Test _get_privacy_mode_name returns correct preset name."""

    @pytest.mark.parametrize("preset", ["ephemeral", "isolated", "anonymous", "normal", "public"])
    def test_preset_names(self, preset):
        f = _make_feature(preset)
        assert f._get_privacy_mode_name() == preset

    def test_no_privacy_agent_returns_normal(self):
        f = _make_feature("normal")
        f.agent.privacy_agent = None
        assert f._get_privacy_mode_name() == "normal"


# ---------------------------------------------------------------------------
# Biometric warning
# ---------------------------------------------------------------------------

class TestBiometricWarning:

    def test_biometric_warning_content(self):
        msg = VoiceFeature.biometric_warning()
        assert "biometric" in msg.lower()
        assert "third-party" in msg.lower()
        assert "voice data" in msg.lower()


# ---------------------------------------------------------------------------
# list_voices privacy filtering
# ---------------------------------------------------------------------------

class TestListVoicesPrivacy:

    @pytest.mark.asyncio
    async def test_ephemeral_filters_cloud_voices(self):
        f = _make_feature("ephemeral", local_tts=True, cloud_tts=True)
        result = await f.list_voices()
        providers = {v["provider"] for v in result["voices"]}
        assert "openai" not in providers
        assert "piper" in providers

    @pytest.mark.asyncio
    async def test_normal_shows_all_voices(self):
        f = _make_feature("normal", local_tts=True, cloud_tts=True)
        result = await f.list_voices()
        providers = {v["provider"] for v in result["voices"]}
        assert "openai" in providers
        assert "piper" in providers

    @pytest.mark.asyncio
    async def test_isolated_filters_cloud_voices(self):
        f = _make_feature("isolated", local_tts=True, cloud_tts=True)
        result = await f.list_voices()
        providers = {v["provider"] for v in result["voices"]}
        assert "openai" not in providers
