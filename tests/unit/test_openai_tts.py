"""
Unit tests for OpenAI TTS provider.

Tests voice listing, availability checks, synthesis with mocked client,
streaming, format mapping, and config handling.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.voice.openai_tts import OpenAITTSProvider
from kestrel_sovereign.voice.base import VoiceInfo


# ---------------------------------------------------------------------------
# Voice listing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_voices_returns_six():
    provider = OpenAITTSProvider()
    voices = await provider.list_voices()
    assert len(voices) == 6
    assert all(isinstance(v, VoiceInfo) for v in voices)


@pytest.mark.asyncio
async def test_list_voices_metadata():
    provider = OpenAITTSProvider()
    voices = await provider.list_voices()
    voice_map = {v.voice_id: v for v in voices}

    assert voice_map["nova"].gender == "feminine"
    assert voice_map["nova"].provider == "openai"
    assert voice_map["echo"].gender == "masculine"
    assert voice_map["alloy"].gender == "neutral"
    assert voice_map["shimmer"].gender == "feminine"
    assert voice_map["onyx"].gender == "masculine"
    assert voice_map["fable"].gender == "neutral"


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_is_available_with_api_key():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        provider = OpenAITTSProvider()
        assert await provider.is_available() is True


@pytest.mark.asyncio
async def test_is_available_without_api_key():
    with patch.dict("os.environ", {}, clear=True):
        provider = OpenAITTSProvider()
        assert await provider.is_available() is False


# ---------------------------------------------------------------------------
# Provider attributes
# ---------------------------------------------------------------------------

def test_provider_name():
    provider = OpenAITTSProvider()
    assert provider.name == "openai"


def test_provider_is_not_local():
    provider = OpenAITTSProvider()
    assert provider.is_local is False


# ---------------------------------------------------------------------------
# Format mapping
# ---------------------------------------------------------------------------

def test_format_map_contains_all_formats():
    assert OpenAITTSProvider.FORMAT_MAP == {
        "opus": "opus",
        "mp3": "mp3",
        "wav": "wav",
        "pcm": "pcm",
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_default_config():
    provider = OpenAITTSProvider()
    assert provider._tts_model == "tts-1-hd"
    assert provider._default_voice == "nova"


def test_custom_config():
    config = {"tts_model": "tts-1", "default_voice": "alloy"}
    provider = OpenAITTSProvider(config=config)
    assert provider._tts_model == "tts-1"
    assert provider._default_voice == "alloy"


# ---------------------------------------------------------------------------
# Synthesis with mocked client
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_synthesize_returns_audio_bytes():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        provider = OpenAITTSProvider()

    mock_response = MagicMock()
    mock_response.content = b"fake-audio-data"
    provider._client = AsyncMock()
    provider._client.audio.speech.create = AsyncMock(return_value=mock_response)

    result = await provider.synthesize("Hello world", voice_id="nova")

    assert result == b"fake-audio-data"
    provider._client.audio.speech.create.assert_called_once_with(
        model="tts-1-hd",
        voice="nova",
        input="Hello world",
        response_format="opus",
    )


@pytest.mark.asyncio
async def test_synthesize_with_model_override():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        provider = OpenAITTSProvider()

    mock_response = MagicMock()
    mock_response.content = b"hd-audio"
    provider._client = AsyncMock()
    provider._client.audio.speech.create = AsyncMock(return_value=mock_response)

    result = await provider.synthesize("Test", voice_id="echo", model="tts-1", output_format="mp3")

    assert result == b"hd-audio"
    provider._client.audio.speech.create.assert_called_once_with(
        model="tts-1",
        voice="echo",
        input="Test",
        response_format="mp3",
    )


@pytest.mark.asyncio
async def test_synthesize_raises_without_client():
    with patch.dict("os.environ", {}, clear=True):
        provider = OpenAITTSProvider()

    with pytest.raises(RuntimeError, match="OpenAI API key not configured"):
        await provider.synthesize("Hello", voice_id="nova")


# ---------------------------------------------------------------------------
# Streaming synthesis with mocked client
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_synthesize_stream_yields_chunks():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        provider = OpenAITTSProvider()

    # Build a mock async context manager that yields chunks
    async def mock_iter_bytes(chunk_size=4096):
        yield b"chunk-1"
        yield b"chunk-2"
        yield b"chunk-3"

    mock_streaming_response = AsyncMock()
    mock_streaming_response.iter_bytes = mock_iter_bytes
    mock_streaming_response.__aenter__ = AsyncMock(return_value=mock_streaming_response)
    mock_streaming_response.__aexit__ = AsyncMock(return_value=False)

    provider._client = MagicMock()
    provider._client.audio.speech.with_streaming_response.create = MagicMock(
        return_value=mock_streaming_response
    )

    chunks = []
    async for chunk in provider.synthesize_stream("Hello streaming", voice_id="nova"):
        chunks.append(chunk)

    assert chunks == [b"chunk-1", b"chunk-2", b"chunk-3"]


@pytest.mark.asyncio
async def test_synthesize_stream_raises_without_client():
    with patch.dict("os.environ", {}, clear=True):
        provider = OpenAITTSProvider()

    with pytest.raises(RuntimeError, match="OpenAI API key not configured"):
        async for _ in provider.synthesize_stream("Hello", voice_id="nova"):
            pass


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------

def test_models_list():
    assert "tts-1" in OpenAITTSProvider.MODELS
    assert "tts-1-hd" in OpenAITTSProvider.MODELS
