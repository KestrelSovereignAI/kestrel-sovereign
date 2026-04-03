"""
Unit tests for Deepgram STT provider.

Tests availability checks, batch transcription with mocked client,
and streaming transcription with mocked WebSocket.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_prerecorded_options(**kwargs):
    """Create a mock PrerecordedOptions that stores kwargs as attributes."""
    obj = MagicMock()
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


def _mock_live_options(**kwargs):
    """Create a mock LiveOptions that stores kwargs as attributes."""
    obj = MagicMock()
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


# ---------------------------------------------------------------------------
# Availability tests
# ---------------------------------------------------------------------------

class TestDeepgramAvailability:
    """Test is_available() under various conditions."""

    @pytest.mark.asyncio
    async def test_available_when_sdk_and_key_present(self):
        """Provider is available when SDK installed and API key set."""
        with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}):
            with patch("kestrel_voice_deepgram.deepgram_stt.DEEPGRAM_AVAILABLE", True):
                from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider
                provider = DeepgramSTTProvider()
                assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_unavailable_when_sdk_missing(self):
        """Provider is unavailable when deepgram-sdk is not installed."""
        with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}):
            with patch("kestrel_voice_deepgram.deepgram_stt.DEEPGRAM_AVAILABLE", False):
                from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider
                provider = DeepgramSTTProvider()
                assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_unavailable_when_key_missing(self):
        """Provider is unavailable when DEEPGRAM_API_KEY is not set."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("kestrel_voice_deepgram.deepgram_stt.DEEPGRAM_AVAILABLE", True):
                from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider
                provider = DeepgramSTTProvider()
                assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_unavailable_when_both_missing(self):
        """Provider is unavailable when both SDK and key are missing."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("kestrel_voice_deepgram.deepgram_stt.DEEPGRAM_AVAILABLE", False):
                from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider
                provider = DeepgramSTTProvider()
                assert await provider.is_available() is False


# ---------------------------------------------------------------------------
# Provider metadata tests
# ---------------------------------------------------------------------------

class TestDeepgramMetadata:
    def test_name(self):
        from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider
        provider = DeepgramSTTProvider()
        assert provider.name == "deepgram"

    def test_is_cloud(self):
        from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider
        provider = DeepgramSTTProvider()
        assert provider.is_local is False

    def test_default_config(self):
        from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider
        provider = DeepgramSTTProvider()
        assert provider._model == "nova-2"
        assert provider._language == "en"

    def test_custom_config(self):
        from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider
        provider = DeepgramSTTProvider(config={"model": "nova-2-meeting", "language": "es"})
        assert provider._model == "nova-2-meeting"
        assert provider._language == "es"

    def test_models_list(self):
        from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider
        assert "nova-2" in DeepgramSTTProvider.MODELS
        assert len(DeepgramSTTProvider.MODELS) == 4


# ---------------------------------------------------------------------------
# Batch transcription tests
# ---------------------------------------------------------------------------

class TestDeepgramBatchTranscription:
    @pytest.mark.asyncio
    async def test_transcribe_returns_text(self):
        """Batch transcription returns transcript from mocked Deepgram client."""
        from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider

        mock_alternative = MagicMock()
        mock_alternative.transcript = "Hello world"

        mock_channel = MagicMock()
        mock_channel.alternatives = [mock_alternative]

        mock_results = MagicMock()
        mock_results.channels = [mock_channel]

        mock_response = MagicMock()
        mock_response.results = mock_results

        mock_rest_v1 = AsyncMock()
        mock_rest_v1.transcribe_file = AsyncMock(return_value=mock_response)

        mock_rest = MagicMock()
        mock_rest.v = MagicMock(return_value=mock_rest_v1)

        mock_listen = MagicMock()
        mock_listen.asyncrest = mock_rest

        mock_client = MagicMock()
        mock_client.listen = mock_listen

        with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}):
            with patch("kestrel_voice_deepgram.deepgram_stt.DEEPGRAM_AVAILABLE", True):
                with patch("kestrel_voice_deepgram.deepgram_stt.DeepgramClient", return_value=mock_client):
                    with patch("kestrel_voice_deepgram.deepgram_stt.PrerecordedOptions", _mock_prerecorded_options):
                        provider = DeepgramSTTProvider()
                        result = await provider.transcribe(b"fake_audio", audio_format="wav")

        assert result == "Hello world"
        mock_rest_v1.transcribe_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_uses_language_override(self):
        """Language parameter overrides default config."""
        from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider

        mock_alternative = MagicMock()
        mock_alternative.transcript = "Hola mundo"

        mock_channel = MagicMock()
        mock_channel.alternatives = [mock_alternative]

        mock_results = MagicMock()
        mock_results.channels = [mock_channel]

        mock_response = MagicMock()
        mock_response.results = mock_results

        mock_rest_v1 = AsyncMock()
        mock_rest_v1.transcribe_file = AsyncMock(return_value=mock_response)

        mock_rest = MagicMock()
        mock_rest.v = MagicMock(return_value=mock_rest_v1)

        mock_listen = MagicMock()
        mock_listen.asyncrest = mock_rest

        mock_client = MagicMock()
        mock_client.listen = mock_listen

        captured_options = {}

        async def capture_call(source, options):
            captured_options["language"] = options.language
            return mock_response

        mock_rest_v1.transcribe_file = capture_call

        with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}):
            with patch("kestrel_voice_deepgram.deepgram_stt.DEEPGRAM_AVAILABLE", True):
                with patch("kestrel_voice_deepgram.deepgram_stt.DeepgramClient", return_value=mock_client):
                    with patch("kestrel_voice_deepgram.deepgram_stt.PrerecordedOptions", _mock_prerecorded_options):
                        provider = DeepgramSTTProvider()
                        await provider.transcribe(b"audio", language="es")

        assert captured_options["language"] == "es"

    @pytest.mark.asyncio
    async def test_transcribe_raises_without_sdk(self):
        """Transcribe raises RuntimeError when SDK not available."""
        from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider

        with patch("kestrel_voice_deepgram.deepgram_stt.DEEPGRAM_AVAILABLE", False):
            provider = DeepgramSTTProvider()
            with pytest.raises(RuntimeError, match="not installed"):
                await provider.transcribe(b"audio")

    @pytest.mark.asyncio
    async def test_transcribe_raises_without_api_key(self):
        """Transcribe raises RuntimeError when API key not set."""
        from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider

        with patch.dict("os.environ", {}, clear=True):
            with patch("kestrel_voice_deepgram.deepgram_stt.DEEPGRAM_AVAILABLE", True):
                provider = DeepgramSTTProvider()
                with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
                    await provider.transcribe(b"audio")


# ---------------------------------------------------------------------------
# Streaming transcription tests
# ---------------------------------------------------------------------------

class TestDeepgramStreamingTranscription:
    @pytest.mark.asyncio
    async def test_transcribe_stream_yields_transcripts(self):
        """Streaming transcription yields text segments from mocked WebSocket."""
        from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider

        event_handlers = {}

        mock_connection = AsyncMock()
        mock_connection.start = AsyncMock(return_value=True)
        mock_connection.finish = AsyncMock()

        def capture_on(event_name, handler):
            event_handlers[event_name] = handler

        mock_connection.on = capture_on
        mock_connection.send = AsyncMock()

        mock_ws = MagicMock()
        mock_ws.v = MagicMock(return_value=mock_connection)

        mock_listen = MagicMock()
        mock_listen.asyncwebsocket = mock_ws

        mock_client = MagicMock()
        mock_client.listen = mock_listen

        patches = [
            patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}),
            patch("kestrel_voice_deepgram.deepgram_stt.DEEPGRAM_AVAILABLE", True),
            patch("kestrel_voice_deepgram.deepgram_stt.DeepgramClient", return_value=mock_client),
            patch("kestrel_voice_deepgram.deepgram_stt.LiveOptions", _mock_live_options),
        ]
        for p in patches:
            p.start()

        try:
            provider = DeepgramSTTProvider()

            async def audio_gen():
                yield b"chunk1"
                yield b"chunk2"

            transcripts = []
            stream = provider.transcribe_stream(audio_gen())

            async def drive_stream():
                async for text in stream:
                    transcripts.append(text)

            task = asyncio.create_task(drive_stream())
            await asyncio.sleep(0.05)

            # Verify event handlers were registered
            assert "Results" in event_handlers
            assert "Error" in event_handlers

            # Simulate Deepgram sending partial transcript
            mock_result = MagicMock()
            mock_result.channel.alternatives = [MagicMock(transcript="Hello")]
            mock_result.is_final = False
            await event_handlers["Results"](mock_connection, mock_result)

            # Simulate final transcript
            mock_result2 = MagicMock()
            mock_result2.channel.alternatives = [MagicMock(transcript="Hello world")]
            mock_result2.is_final = True
            await event_handlers["Results"](mock_connection, mock_result2)

            await asyncio.sleep(0.05)

            # Signal end via error handler (puts None in queue)
            await event_handlers["Error"](mock_connection, "stream ended")

            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            assert "Hello" in transcripts
            assert "Hello world" in transcripts
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_transcribe_stream_raises_without_sdk(self):
        """Stream raises RuntimeError when SDK not available."""
        from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider

        with patch("kestrel_voice_deepgram.deepgram_stt.DEEPGRAM_AVAILABLE", False):
            provider = DeepgramSTTProvider()

            async def audio_gen():
                yield b"chunk"

            with pytest.raises(RuntimeError, match="not installed"):
                async for _ in provider.transcribe_stream(audio_gen()):
                    pass

    @pytest.mark.asyncio
    async def test_transcribe_stream_raises_without_api_key(self):
        """Stream raises RuntimeError when API key not set."""
        from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider

        with patch.dict("os.environ", {}, clear=True):
            with patch("kestrel_voice_deepgram.deepgram_stt.DEEPGRAM_AVAILABLE", True):
                provider = DeepgramSTTProvider()

                async def audio_gen():
                    yield b"chunk"

                with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
                    async for _ in provider.transcribe_stream(audio_gen()):
                        pass

    @pytest.mark.asyncio
    async def test_transcribe_stream_raises_on_connection_failure(self):
        """Stream raises RuntimeError when WebSocket connection fails."""
        from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider

        mock_connection = AsyncMock()
        mock_connection.start = AsyncMock(return_value=False)
        mock_connection.on = MagicMock()

        mock_ws = MagicMock()
        mock_ws.v = MagicMock(return_value=mock_connection)

        mock_listen = MagicMock()
        mock_listen.asyncwebsocket = mock_ws

        mock_client = MagicMock()
        mock_client.listen = mock_listen

        with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}):
            with patch("kestrel_voice_deepgram.deepgram_stt.DEEPGRAM_AVAILABLE", True):
                with patch("kestrel_voice_deepgram.deepgram_stt.DeepgramClient", return_value=mock_client):
                    with patch("kestrel_voice_deepgram.deepgram_stt.LiveOptions", _mock_live_options):
                        provider = DeepgramSTTProvider()

                        async def audio_gen():
                            yield b"chunk"

                        with pytest.raises(RuntimeError, match="Failed to start"):
                            async for _ in provider.transcribe_stream(audio_gen()):
                                pass


# ---------------------------------------------------------------------------
# Registry integration test
# ---------------------------------------------------------------------------

class TestDeepgramRegistryIntegration:
    @pytest.mark.asyncio
    async def test_registry_creates_deepgram_provider(self):
        """VoiceProviderRegistry._create_stt_provider returns DeepgramSTTProvider."""
        from kestrel_sovereign.voice import VoiceProviderRegistry

        config = {"deepgram": {"model": "nova-2-meeting", "language": "fr"}}
        registry = VoiceProviderRegistry(config=config)
        provider = registry._create_stt_provider("deepgram")

        if provider is None:
            pytest.skip("deepgram-sdk not installed")

        assert provider.name == "deepgram"
        assert provider._model == "nova-2-meeting"
        assert provider._language == "fr"

    @pytest.mark.asyncio
    async def test_registry_initialize_registers_deepgram(self):
        """Registry initialize() registers deepgram when available."""
        from kestrel_sovereign.voice import VoiceProviderRegistry
        from kestrel_sovereign.voice.deepgram_stt import DeepgramSTTProvider

        config = {
            "stt_provider_priority": ["deepgram"],
            "deepgram": {"model": "nova-2"},
        }
        registry = VoiceProviderRegistry(config=config)

        with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}):
            with patch("kestrel_voice_deepgram.deepgram_stt.DEEPGRAM_AVAILABLE", True):
                await registry.initialize()

        assert "deepgram" in registry.list_stt_providers()
        provider = registry.get_stt("deepgram")
        assert isinstance(provider, DeepgramSTTProvider)
