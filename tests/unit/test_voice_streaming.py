"""Unit tests for streaming TTS — sentence splitting, stream tap, and endpoint modes."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.voice.base import split_sentences
from kestrel_sovereign.voice.stream_tap import AgentStreamTap


# ------------------------------------------------------------------
# split_sentences
# ------------------------------------------------------------------


class TestSplitSentences:
    def test_simple_sentences(self):
        result = split_sentences("Hello world. How are you? I am fine!")
        assert result == ["Hello world.", "How are you?", "I am fine!"]

    def test_single_sentence(self):
        assert split_sentences("Hello world.") == ["Hello world."]

    def test_no_punctuation(self):
        assert split_sentences("Hello world") == ["Hello world"]

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_whitespace_only(self):
        assert split_sentences("   ") == []

    def test_none_input(self):
        assert split_sentences(None) == []

    def test_multiple_spaces_between_sentences(self):
        result = split_sentences("First sentence.   Second sentence.")
        assert result == ["First sentence.", "Second sentence."]

    def test_exclamation_and_question(self):
        result = split_sentences("Really? Yes! Let's go.")
        assert result == ["Really?", "Yes!", "Let's go."]

    def test_abbreviations_not_split(self):
        # Abbreviation without trailing space shouldn't split
        result = split_sentences("Dr.Smith went home.")
        assert result == ["Dr.Smith went home."]

    def test_trailing_whitespace(self):
        result = split_sentences("  Hello world.  ")
        assert result == ["Hello world."]

    def test_newlines_between_sentences(self):
        # Newlines act as whitespace
        result = split_sentences("First.\nSecond.")
        assert result == ["First.", "Second."]


# ------------------------------------------------------------------
# AgentStreamTap
# ------------------------------------------------------------------


class TestAgentStreamTap:
    def setup_method(self):
        AgentStreamTap.reset()

    def teardown_method(self):
        AgentStreamTap.reset()

    def test_singleton(self):
        t1 = AgentStreamTap.get_instance()
        t2 = AgentStreamTap.get_instance()
        assert t1 is t2

    def test_reset(self):
        t1 = AgentStreamTap.get_instance()
        AgentStreamTap.reset()
        t2 = AgentStreamTap.get_instance()
        assert t1 is not t2

    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self):
        tap = AgentStreamTap.get_instance()
        tap.register("req-1")

        # Publish chunks then finish in background
        async def producer():
            await asyncio.sleep(0.01)
            await tap.publish("req-1", "Hello ")
            await tap.publish("req-1", "world.")
            await tap.finish("req-1")

        chunks = []
        task = asyncio.create_task(producer())
        async for chunk in tap.subscribe("req-1"):
            chunks.append(chunk)
        await task

        assert chunks == ["Hello ", "world."]

    @pytest.mark.asyncio
    async def test_subscribe_unregistered_returns_nothing(self):
        tap = AgentStreamTap.get_instance()
        chunks = []
        async for chunk in tap.subscribe("nonexistent"):
            chunks.append(chunk)
        assert chunks == []

    def test_has_stream(self):
        tap = AgentStreamTap.get_instance()
        assert not tap.has_stream("req-1")
        tap.register("req-1")
        assert tap.has_stream("req-1")
        tap.unregister("req-1")
        assert not tap.has_stream("req-1")

    @pytest.mark.asyncio
    async def test_subscribe_timeout(self):
        """Subscribe should stop after timeout if no chunks arrive."""
        tap = AgentStreamTap.get_instance()
        tap.register("req-timeout")

        chunks = []
        async for chunk in tap.subscribe("req-timeout", timeout=0.1):
            chunks.append(chunk)
        assert chunks == []


# ------------------------------------------------------------------
# Streaming TTS endpoint
# ------------------------------------------------------------------


def _make_voice_feature(voice_config=None):
    """Build a mock VoiceFeature with controllable behaviour."""
    from kestrel_sovereign.voice.base import VoiceConfig, VoiceInfo

    vf = MagicMock()
    vf._voice_config = voice_config or VoiceConfig()
    vf.list_voices = AsyncMock(return_value={"voices": [], "count": 0})

    # TTS provider mock
    tts = AsyncMock()
    tts.synthesize = AsyncMock(return_value=b"\x00audio")
    tts.list_voices = AsyncMock(return_value=[
        VoiceInfo(voice_id="nova", name="Nova", provider="openai"),
    ])

    async def _stream_chunks(*a, **kw):
        yield b"\x00chunk1"
        yield b"\x01chunk2"

    tts.synthesize_stream = _stream_chunks
    tts.name = "openai"
    vf._get_tts_provider = AsyncMock(return_value=tts)

    # STT provider mock
    stt = AsyncMock()
    stt.transcribe = AsyncMock(return_value="hello")
    stt.name = "openai"
    vf._get_stt_provider = AsyncMock(return_value=stt)

    # stream_speak mock for agent-response mode
    async def _mock_stream_speak(**kw):
        yield b"\x00agent-audio-1"
        yield b"\x01agent-audio-2"

    vf.stream_speak = _mock_stream_speak

    return vf


def _make_agent(voice_feature=None):
    agent = MagicMock()
    features = {}
    if voice_feature:
        features["VoiceFeature"] = voice_feature
    agent.features = features
    agent.identity = MagicMock()
    agent.identity.voice_config = None
    return agent


def _prepare_app(agent):
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
    }
    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    return app, original


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


class TestTTSStreamFullTextMode:
    """Mode 1: body.text provided — splits into sentences and streams."""

    def test_stream_returns_chunked_audio(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                from fastapi.testclient import TestClient
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/tts/stream",
                        json={"text": "Hello world. How are you?", "format": "opus"},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "audio/opus"
            assert b"chunk1" in resp.content
            assert b"chunk2" in resp.content
        finally:
            _restore_app(app, orig)

    def test_stream_correct_headers(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                from fastapi.testclient import TestClient
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/tts/stream",
                        json={"text": "Test sentence."},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.headers.get("cache-control") == "no-cache"
            assert resp.headers.get("x-accel-buffering") == "no"
        finally:
            _restore_app(app, orig)

    def test_stream_mp3_content_type(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                from fastapi.testclient import TestClient
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/tts/stream",
                        json={"text": "Hello.", "format": "mp3"},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.headers["content-type"] == "audio/mpeg"
        finally:
            _restore_app(app, orig)


class TestTTSStreamAgentMode:
    """Mode 2: body.request_id provided — taps into active agent stream."""

    def test_agent_mode_returns_audio_from_stream_speak(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                from fastapi.testclient import TestClient
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/tts/stream",
                        json={"request_id": "test-req-123", "format": "opus"},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 200
            assert b"agent-audio-1" in resp.content
            assert b"agent-audio-2" in resp.content
        finally:
            _restore_app(app, orig)


class TestTTSStreamValidation:
    """Validation: neither text nor request_id provided."""

    def test_returns_400_when_no_text_or_request_id(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                from fastapi.testclient import TestClient
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/tts/stream",
                        json={},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 400
            assert "text" in resp.json()["detail"].lower() or "request_id" in resp.json()["detail"]
        finally:
            _restore_app(app, orig)

    def test_returns_403_when_privacy_blocks_tts(self):
        from kestrel_sovereign.features.voice.feature import VoicePrivacyError

        vf = _make_voice_feature()
        vf._get_tts_provider = AsyncMock(
            side_effect=VoicePrivacyError("Blocked by privacy mode")
        )
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                from fastapi.testclient import TestClient
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/tts/stream",
                        json={"text": "Hello"},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 403
        finally:
            _restore_app(app, orig)
