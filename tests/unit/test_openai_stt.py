"""
Unit tests for OpenAI STT provider (Whisper API adapter).

Tests availability checks, transcription with mocked OpenAI client,
file size limit handling, format validation, and streaming.
"""
import io
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.voice.openai_stt import (
    MAX_FILE_SIZE,
    SUPPORTED_FORMATS,
    OpenAISTTProvider,
)


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

class TestAvailability:
    @pytest.mark.asyncio
    async def test_available_when_key_set(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        provider = OpenAISTTProvider()
        assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_unavailable_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAISTTProvider()
        assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_unavailable_when_key_empty(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        provider = OpenAISTTProvider()
        assert await provider.is_available() is False


# ---------------------------------------------------------------------------
# Provider metadata
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_name(self):
        assert OpenAISTTProvider.name == "openai"

    def test_is_cloud(self):
        assert OpenAISTTProvider.is_local is False

    def test_default_model(self):
        provider = OpenAISTTProvider()
        assert provider._model == "whisper-1"

    def test_custom_model_from_config(self):
        provider = OpenAISTTProvider(config={"stt_model": "whisper-2"})
        assert provider._model == "whisper-2"


# ---------------------------------------------------------------------------
# Format validation
# ---------------------------------------------------------------------------

class TestFormatValidation:
    @pytest.mark.asyncio
    async def test_rejects_unsupported_format(self):
        provider = OpenAISTTProvider()
        with pytest.raises(ValueError, match="Unsupported audio format"):
            await provider.transcribe(b"audio", audio_format="aiff")

    def test_supported_formats_list(self):
        expected = {"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm", "opus"}
        assert set(SUPPORTED_FORMATS) == expected


# ---------------------------------------------------------------------------
# Transcription (mocked OpenAI client)
# ---------------------------------------------------------------------------

def _make_mock_client(transcription_result="Hello world"):
    """Create a mock AsyncOpenAI client that returns a fixed transcription."""
    mock_client = AsyncMock()
    mock_client.audio.transcriptions.create = AsyncMock(return_value=transcription_result)
    return mock_client


class TestTranscribe:
    @pytest.mark.asyncio
    async def test_basic_transcription(self):
        provider = OpenAISTTProvider()
        mock_client = _make_mock_client("Hello world")

        with patch("kestrel_sovereign.voice.openai_stt.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_client
            result = await provider.transcribe(b"fake-audio-data", audio_format="wav")

        assert result == "Hello world"
        mock_client.audio.transcriptions.create.assert_called_once()
        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == "whisper-1"
        assert call_kwargs["response_format"] == "text"

    @pytest.mark.asyncio
    async def test_transcription_with_language_hint(self):
        provider = OpenAISTTProvider()
        mock_client = _make_mock_client("Bonjour")

        with patch("kestrel_sovereign.voice.openai_stt.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_client
            result = await provider.transcribe(
                b"french-audio", language="fr", audio_format="mp3",
            )

        assert result == "Bonjour"
        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["language"] == "fr"

    @pytest.mark.asyncio
    async def test_transcription_no_language_omits_param(self):
        provider = OpenAISTTProvider()
        mock_client = _make_mock_client("Hello")

        with patch("kestrel_sovereign.voice.openai_stt.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_client
            await provider.transcribe(b"audio", audio_format="wav")

        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert "language" not in call_kwargs

    @pytest.mark.asyncio
    async def test_file_tuple_has_correct_extension(self):
        provider = OpenAISTTProvider()
        mock_client = _make_mock_client("text")

        with patch("kestrel_sovereign.voice.openai_stt.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_client
            await provider.transcribe(b"audio", audio_format="flac")

        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        filename, _ = call_kwargs["file"]
        assert filename == "audio.flac"

    @pytest.mark.asyncio
    async def test_opus_uses_ogg_extension(self):
        provider = OpenAISTTProvider()
        mock_client = _make_mock_client("text")

        with patch("kestrel_sovereign.voice.openai_stt.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_client
            await provider.transcribe(b"audio", audio_format="opus")

        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        filename, _ = call_kwargs["file"]
        assert filename == "audio.ogg"


# ---------------------------------------------------------------------------
# File size limit handling
# ---------------------------------------------------------------------------

class TestFileSizeLimit:
    @pytest.mark.asyncio
    async def test_small_file_single_call(self):
        provider = OpenAISTTProvider()
        mock_client = _make_mock_client("single chunk")

        with patch("kestrel_sovereign.voice.openai_stt.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_client
            audio = b"x" * (MAX_FILE_SIZE - 1)
            result = await provider.transcribe(audio, audio_format="wav")

        assert result == "single chunk"
        assert mock_client.audio.transcriptions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_large_file_split_into_chunks(self):
        provider = OpenAISTTProvider()
        mock_client = _make_mock_client("part")

        with patch("kestrel_sovereign.voice.openai_stt.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_client
            # Just over 2 chunks worth
            audio = b"x" * (MAX_FILE_SIZE * 2 + 100)
            result = await provider.transcribe(audio, audio_format="wav")

        assert mock_client.audio.transcriptions.create.call_count == 3
        assert result == "part part part"

    @pytest.mark.asyncio
    async def test_exact_limit_single_call(self):
        provider = OpenAISTTProvider()
        mock_client = _make_mock_client("ok")

        with patch("kestrel_sovereign.voice.openai_stt.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_client
            audio = b"x" * MAX_FILE_SIZE
            await provider.transcribe(audio, audio_format="wav")

        assert mock_client.audio.transcriptions.create.call_count == 1


# ---------------------------------------------------------------------------
# Streaming transcription
# ---------------------------------------------------------------------------

class TestTranscribeStream:
    @pytest.mark.asyncio
    async def test_stream_yields_partial_results(self):
        provider = OpenAISTTProvider()
        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return f"partial result {call_count}"

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = mock_create

        async def audio_gen():
            # Yield enough data to trigger at least one flush (>1 MB)
            chunk_size = 512 * 1024  # 512 KB
            for _ in range(4):
                yield b"x" * chunk_size

        with patch("kestrel_sovereign.voice.openai_stt.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_client
            results = []
            async for text in provider.transcribe_stream(audio_gen()):
                results.append(text)

        assert len(results) >= 1
        assert all(r.startswith("partial result") for r in results)

    @pytest.mark.asyncio
    async def test_stream_final_flush(self):
        """Small audio that never hits the flush threshold still gets transcribed."""
        provider = OpenAISTTProvider()

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value="final")

        async def audio_gen():
            yield b"small chunk"

        with patch("kestrel_sovereign.voice.openai_stt.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_client
            results = []
            async for text in provider.transcribe_stream(audio_gen()):
                results.append(text)

        assert results == ["final"]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    @pytest.mark.asyncio
    async def test_registry_creates_openai_stt(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from kestrel_sovereign.voice import VoiceProviderRegistry

        config = {"stt_provider_priority": ["openai"]}
        registry = VoiceProviderRegistry(config=config)
        await registry.initialize()
        provider = registry.get_stt("openai")
        assert provider is not None
        assert provider.name == "openai"

    @pytest.mark.asyncio
    async def test_registry_skips_openai_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from kestrel_sovereign.voice import VoiceProviderRegistry

        config = {"stt_provider_priority": ["openai"]}
        registry = VoiceProviderRegistry(config=config)
        await registry.initialize()
        assert registry.get_stt("openai") is None
