"""
Integration tests for OpenAI voice providers — real API calls.

Requires OPENAI_API_KEY. Skipped automatically when key is missing.
"""
import os
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    ),
]


@pytest.fixture
def tts_provider():
    from kestrel_sovereign.voice.openai_tts import OpenAITTSProvider
    return OpenAITTSProvider(config={"tts_model": "tts-1", "default_voice": "nova"})


@pytest.fixture
def stt_provider():
    from kestrel_sovereign.voice.openai_stt import OpenAISTTProvider
    return OpenAISTTProvider(config={"stt_model": "whisper-1"})


class TestOpenAITTSReal:
    """Real OpenAI TTS synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_returns_audio_bytes(self, tts_provider):
        audio = await tts_provider.synthesize(
            "Hello world", voice_id="nova", output_format="mp3",
        )
        assert isinstance(audio, bytes)
        assert len(audio) > 1000  # MP3 should be at least a few KB

    @pytest.mark.asyncio
    async def test_synthesize_stream_yields_chunks(self, tts_provider):
        chunks = []
        async for chunk in tts_provider.synthesize_stream(
            "This is a streaming test.", voice_id="nova", output_format="mp3",
        ):
            chunks.append(chunk)
        assert len(chunks) >= 1
        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes > 500

    @pytest.mark.asyncio
    async def test_all_voices_produce_audio(self, tts_provider):
        voices = await tts_provider.list_voices()
        assert len(voices) == 6
        for voice in voices:
            audio = await tts_provider.synthesize(
                "Test", voice_id=voice.voice_id, output_format="mp3",
            )
            assert len(audio) > 500, f"Voice {voice.voice_id} produced too little audio"

    @pytest.mark.asyncio
    async def test_is_available(self, tts_provider):
        assert await tts_provider.is_available() is True


class TestOpenAISTTReal:
    """Real OpenAI Whisper transcription."""

    @pytest.mark.asyncio
    async def test_transcribe_tts_output(self, tts_provider, stt_provider):
        """TTS → STT round-trip: synthesize then transcribe, verify text matches."""
        audio = await tts_provider.synthesize(
            "The quick brown fox jumps over the lazy dog",
            voice_id="nova",
            output_format="mp3",
        )
        transcript = await stt_provider.transcribe(audio, audio_format="mp3")
        # Whisper should get the gist right
        transcript_lower = transcript.lower()
        assert "fox" in transcript_lower
        assert "dog" in transcript_lower

    @pytest.mark.asyncio
    async def test_transcribe_with_language_hint(self, tts_provider, stt_provider):
        audio = await tts_provider.synthesize(
            "Hello, how are you today?",
            voice_id="nova",
            output_format="mp3",
        )
        transcript = await stt_provider.transcribe(
            audio, language="en", audio_format="mp3",
        )
        assert len(transcript) > 5
        assert "hello" in transcript.lower()

    @pytest.mark.asyncio
    async def test_is_available(self, stt_provider):
        assert await stt_provider.is_available() is True
