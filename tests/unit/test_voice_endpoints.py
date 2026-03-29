"""Unit tests for the voice HTTP endpoints (endpoints/voice.py)."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_voice_feature(voice_config=None):
    """Build a mock VoiceFeature with controllable behaviour."""
    from kestrel_sovereign.voice.base import VoiceConfig, VoiceInfo

    vf = MagicMock()
    vf._voice_config = voice_config or VoiceConfig()
    vf.list_voices = AsyncMock(return_value={
        "voices": [
            {"voice_id": "nova", "name": "Nova", "provider": "openai", "language": "en", "gender": "feminine", "preview_url": ""},
            {"voice_id": "echo", "name": "Echo", "provider": "openai", "language": "en", "gender": "masculine", "preview_url": ""},
        ],
        "count": 2,
    })

    # TTS provider mock
    tts = AsyncMock()
    tts.synthesize = AsyncMock(return_value=b"\x00\x01\x02audio-bytes")
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
    stt.transcribe = AsyncMock(return_value="hello world")
    stt.name = "openai"
    vf._get_stt_provider = AsyncMock(return_value=stt)

    return vf


def _make_agent(voice_feature=None):
    """Build a mock agent with optional VoiceFeature."""
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


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestListVoices:
    def test_list_voices_returns_voices(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.get("/voice/voices", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] == 2
            assert len(data["voices"]) == 2
        finally:
            _restore_app(app, orig)

    def test_list_voices_with_provider_filter(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.get("/voice/voices?provider=openai", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 200
            vf.list_voices.assert_called_once_with(provider="openai")
        finally:
            _restore_app(app, orig)

    def test_list_voices_503_when_no_voice_feature(self):
        agent = _make_agent(voice_feature=None)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.get("/voice/voices", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 503
        finally:
            _restore_app(app, orig)


class TestGetVoiceConfig:
    def test_get_config_returns_current_config(self):
        from kestrel_sovereign.voice.base import VoiceConfig
        cfg = VoiceConfig(tts_provider="openai", tts_voice_id="nova", output_format="mp3")
        vf = _make_voice_feature(voice_config=cfg)
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.get("/voice/config", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["tts_provider"] == "openai"
            assert data["tts_voice_id"] == "nova"
            assert data["output_format"] == "mp3"
        finally:
            _restore_app(app, orig)


class TestSetVoiceConfig:
    def test_set_config_updates_and_returns(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/config",
                        json={"tts_provider": "openai", "tts_voice_id": "nova", "tts_model": "tts-1-hd"},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["config"]["tts_provider"] == "openai"
            assert data["config"]["tts_voice_id"] == "nova"
        finally:
            _restore_app(app, orig)

    def test_set_config_persists_to_identity(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    client.post(
                        "/voice/config",
                        json={"tts_provider": "piper", "tts_voice_id": "lessac"},
                        headers={"X-API-Key": "test-key"},
                    )
            # Identity should have been updated
            assert agent.identity.voice_config is not None
            assert agent.identity.voice_config["tts_provider"] == "piper"
        finally:
            _restore_app(app, orig)


class TestTTS:
    def test_tts_returns_audio_bytes_with_correct_content_type(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/tts",
                        json={"text": "Hello world", "format": "opus"},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "audio/opus"
            assert len(resp.content) > 0
        finally:
            _restore_app(app, orig)

    def test_tts_mp3_content_type(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/tts",
                        json={"text": "Hello", "format": "mp3"},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "audio/mpeg"
        finally:
            _restore_app(app, orig)

    def test_tts_wav_content_type(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/tts",
                        json={"text": "Hello", "format": "wav"},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "audio/wav"
        finally:
            _restore_app(app, orig)

    def test_tts_503_when_no_provider(self):
        vf = _make_voice_feature()
        vf._get_tts_provider = AsyncMock(side_effect=Exception("No TTS provider available."))
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/tts",
                        json={"text": "Hello"},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 503
        finally:
            _restore_app(app, orig)


class TestTTSStream:
    def test_tts_stream_returns_chunked_audio(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/tts/stream",
                        json={"text": "Hello world", "format": "opus"},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "audio/opus"
            # Should have received the concatenation of both chunks
            assert b"chunk1" in resp.content
            assert b"chunk2" in resp.content
        finally:
            _restore_app(app, orig)

    def test_tts_stream_correct_headers(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/tts/stream",
                        json={"text": "Test"},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.headers.get("cache-control") == "no-cache"
        finally:
            _restore_app(app, orig)


class TestSTT:
    def test_stt_transcribes_uploaded_audio(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/stt",
                        files={"file": ("recording.opus", b"\x00\x01audio", "audio/opus")},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 200
            data = resp.json()
            assert data["text"] == "hello world"
        finally:
            _restore_app(app, orig)

    def test_stt_infers_format_from_filename(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/stt",
                        files={"file": ("speech.mp3", b"\x00audio", "audio/mpeg")},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 200
            # Verify the STT provider was called with mp3 format
            stt = (await_result := vf._get_stt_provider.return_value)
            stt.transcribe.assert_called_once()
            call_kwargs = stt.transcribe.call_args
            assert call_kwargs[1].get("audio_format", call_kwargs[0][2] if len(call_kwargs[0]) > 2 else None) == "mp3" or \
                "mp3" in str(call_kwargs)
        finally:
            _restore_app(app, orig)

    def test_stt_empty_file_returns_400(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/stt",
                        files={"file": ("empty.wav", b"", "audio/wav")},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 400
        finally:
            _restore_app(app, orig)

    def test_stt_with_language_param(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/stt?language=es",
                        files={"file": ("recording.opus", b"\x00audio", "audio/opus")},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 200
            data = resp.json()
            assert data["language"] == "es"
        finally:
            _restore_app(app, orig)

    def test_stt_503_when_no_provider(self):
        vf = _make_voice_feature()
        vf._get_stt_provider = AsyncMock(side_effect=Exception("No STT provider available."))
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.post(
                        "/voice/stt",
                        files={"file": ("test.wav", b"\x00audio", "audio/wav")},
                        headers={"X-API-Key": "test-key"},
                    )
            assert resp.status_code == 503
        finally:
            _restore_app(app, orig)


class TestAuthMiddleware:
    """Verify that voice endpoints require authentication."""

    def test_voice_endpoints_require_auth(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    # No auth header — should be rejected
                    assert client.get("/voice/voices").status_code == 401
                    assert client.get("/voice/config").status_code == 401
                    assert client.post("/voice/config", json={}).status_code == 401
                    assert client.post("/voice/tts", json={"text": "hi"}).status_code == 401
                    assert client.post("/voice/tts/stream", json={"text": "hi"}).status_code == 401
                    assert client.post(
                        "/voice/stt",
                        files={"file": ("t.wav", b"\x00", "audio/wav")},
                    ).status_code == 401
        finally:
            _restore_app(app, orig)
