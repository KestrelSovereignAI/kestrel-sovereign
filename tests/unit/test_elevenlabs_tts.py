"""Unit tests for ElevenLabsTTSProvider."""
import importlib
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure ELEVENLABS_API_KEY is unset unless explicitly set in a test."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Availability tests
# ---------------------------------------------------------------------------

class TestIsAvailable:
    """Tests for is_available() checking package + API key."""

    @pytest.mark.asyncio
    async def test_available_when_package_and_key_present(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider
            provider = ElevenLabsTTSProvider()
            assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_unavailable_when_package_missing(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        with patch("importlib.util.find_spec", return_value=None):
            from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider
            provider = ElevenLabsTTSProvider()
            assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_unavailable_when_key_missing(self):
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider
            provider = ElevenLabsTTSProvider()
            assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_unavailable_when_both_missing(self):
        with patch("importlib.util.find_spec", return_value=None):
            from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider
            provider = ElevenLabsTTSProvider()
            assert await provider.is_available() is False


# ---------------------------------------------------------------------------
# Output format mapping tests
# ---------------------------------------------------------------------------

class TestFormatMapping:
    """Tests for canonical → ElevenLabs format mapping."""

    def test_mp3_format(self):
        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider
        assert ElevenLabsTTSProvider._resolve_format("mp3") == "mp3_44100_128"

    def test_pcm_format(self):
        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider
        assert ElevenLabsTTSProvider._resolve_format("pcm") == "pcm_24000"

    def test_opus_falls_back_to_mp3(self):
        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider
        assert ElevenLabsTTSProvider._resolve_format("opus") == "mp3_44100_128"

    def test_ulaw_format(self):
        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider
        assert ElevenLabsTTSProvider._resolve_format("ulaw") == "ulaw_8000"

    def test_unknown_format_defaults_to_mp3(self):
        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider
        assert ElevenLabsTTSProvider._resolve_format("flac") == "mp3_44100_128"


# ---------------------------------------------------------------------------
# Voice listing tests
# ---------------------------------------------------------------------------

class TestListVoices:
    """Tests for list_voices() with mocked ElevenLabs client."""

    @pytest.mark.asyncio
    async def test_list_voices_maps_to_voice_info(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider

        mock_voice = MagicMock()
        mock_voice.voice_id = "abc123"
        mock_voice.name = "Rachel"
        mock_voice.labels = {"gender": "feminine", "language": "en"}
        mock_voice.preview_url = "https://example.com/rachel.mp3"

        mock_cloned = MagicMock()
        mock_cloned.voice_id = "clone456"
        mock_cloned.name = "My Clone"
        mock_cloned.labels = {"gender": "masculine"}
        mock_cloned.preview_url = ""

        mock_response = MagicMock()
        mock_response.voices = [mock_voice, mock_cloned]

        mock_client = AsyncMock()
        mock_client.voices.get_all.return_value = mock_response

        provider = ElevenLabsTTSProvider()
        provider._client = mock_client

        voices = await provider.list_voices()

        assert len(voices) == 2
        assert voices[0].voice_id == "abc123"
        assert voices[0].name == "Rachel"
        assert voices[0].provider == "elevenlabs"
        assert voices[0].gender == "feminine"
        assert voices[0].language == "en"
        assert voices[0].preview_url == "https://example.com/rachel.mp3"

        assert voices[1].voice_id == "clone456"
        assert voices[1].name == "My Clone"
        assert voices[1].language == "en"  # default when not in labels

    @pytest.mark.asyncio
    async def test_list_voices_handles_empty_labels(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider

        mock_voice = MagicMock()
        mock_voice.voice_id = "v1"
        mock_voice.name = "Test"
        mock_voice.labels = {}
        mock_voice.preview_url = None

        mock_response = MagicMock()
        mock_response.voices = [mock_voice]

        mock_client = AsyncMock()
        mock_client.voices.get_all.return_value = mock_response

        provider = ElevenLabsTTSProvider()
        provider._client = mock_client

        voices = await provider.list_voices()
        assert voices[0].gender == "neutral"
        assert voices[0].language == "en"
        assert voices[0].preview_url == ""


# ---------------------------------------------------------------------------
# Synthesis tests
# ---------------------------------------------------------------------------

class TestSynthesize:
    """Tests for synthesize() with mocked ElevenLabs client."""

    @pytest.mark.asyncio
    async def test_synthesize_returns_bytes(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider

        mock_client = AsyncMock()
        mock_client.text_to_speech.convert.return_value = b"fake-audio-data"

        provider = ElevenLabsTTSProvider(config={"model": "eleven_turbo_v2_5"})
        provider._client = mock_client

        result = await provider.synthesize("Hello world", voice_id="voice1")

        assert result == b"fake-audio-data"
        mock_client.text_to_speech.convert.assert_called_once_with(
            voice_id="voice1",
            text="Hello world",
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
        )

    @pytest.mark.asyncio
    async def test_synthesize_uses_default_voice_id(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider

        mock_client = AsyncMock()
        mock_client.text_to_speech.convert.return_value = b"audio"

        provider = ElevenLabsTTSProvider(config={"default_voice_id": "default_v"})
        provider._client = mock_client

        await provider.synthesize("Hi", voice_id="")
        call_kwargs = mock_client.text_to_speech.convert.call_args[1]
        assert call_kwargs["voice_id"] == "default_v"

    @pytest.mark.asyncio
    async def test_synthesize_raises_without_voice_id(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider

        provider = ElevenLabsTTSProvider()
        provider._client = AsyncMock()

        with pytest.raises(ValueError, match="voice_id is required"):
            await provider.synthesize("Hello", voice_id="")

    @pytest.mark.asyncio
    async def test_synthesize_collects_async_iterator(self, monkeypatch):
        """When SDK returns an async iterator instead of bytes, collect chunks."""
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider

        async def fake_stream():
            yield b"chunk1"
            yield b"chunk2"

        mock_client = AsyncMock()
        mock_client.text_to_speech.convert.return_value = fake_stream()

        provider = ElevenLabsTTSProvider()
        provider._client = mock_client

        result = await provider.synthesize("Hello", voice_id="v1")
        assert result == b"chunk1chunk2"


class TestSynthesizeStream:
    """Tests for synthesize_stream() with mocked ElevenLabs client."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider

        async def fake_stream():
            yield b"chunk1"
            yield b"chunk2"
            yield b"chunk3"

        mock_client = AsyncMock()
        mock_client.text_to_speech.convert_as_stream.return_value = fake_stream()

        provider = ElevenLabsTTSProvider()
        provider._client = mock_client

        chunks = []
        async for chunk in provider.synthesize_stream("Hello", voice_id="v1"):
            chunks.append(chunk)

        assert chunks == [b"chunk1", b"chunk2", b"chunk3"]

    @pytest.mark.asyncio
    async def test_stream_raises_without_voice_id(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider

        provider = ElevenLabsTTSProvider()
        provider._client = AsyncMock()

        with pytest.raises(ValueError, match="voice_id is required"):
            async for _ in provider.synthesize_stream("Hello", voice_id=""):
                pass


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Tests for provider configuration."""

    def test_default_model(self):
        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider
        provider = ElevenLabsTTSProvider()
        assert provider._model == "eleven_multilingual_v2"

    def test_custom_model_from_config(self):
        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider
        provider = ElevenLabsTTSProvider(config={"model": "eleven_turbo_v2_5"})
        assert provider._model == "eleven_turbo_v2_5"

    def test_provider_attributes(self):
        from kestrel_sovereign.voice.elevenlabs_tts import ElevenLabsTTSProvider
        provider = ElevenLabsTTSProvider()
        assert provider.name == "elevenlabs"
        assert provider.is_local is False
