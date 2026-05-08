"""Unit tests for the WebSocket /voice/chat endpoint (endpoints/voice.py)."""

import asyncio
import json
import sys
import threading
import types
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from kestrel_sovereign.endpoints.voice import (
    FRAME_AUDIO,
    FRAME_JSON,
    encode_audio,
    encode_control,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _decode_frame(data: bytes) -> tuple:
    """Decode a framed message into (frame_type, payload).

    Returns:
        (FRAME_JSON, dict) or (FRAME_AUDIO, bytes)
    """
    frame_type = data[0]
    payload = data[1:]
    if frame_type == FRAME_JSON:
        return frame_type, json.loads(payload)
    return frame_type, payload


@pytest.fixture
def disable_vad():
    """Suppress webrtcvad import in the WS handler.

    The WS handler initializes VAD when ``kestrel_sovereign.voice.vad`` is
    importable. Since #793 moved webrtcvad to core deps it's always present
    in CI — and tests that send synthetic bytes (not real PCM frames) hang
    because VAD never detects ``speech_end``. This fixture forces the
    handler down its no-VAD passthrough path so synthetic-audio tests
    finish in milliseconds.
    """
    vad_key = "kestrel_sovereign.voice.vad"
    saved = sys.modules.pop(vad_key, None)
    sys.modules[vad_key] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        if saved is not None:
            sys.modules[vad_key] = saved
        else:
            sys.modules.pop(vad_key, None)


def _make_voice_feature(cloud_allowed=True, has_stt=True, has_tts=True,
                        stt_streaming=True, privacy_error=None):
    """Build a mock VoiceFeature for WebSocket tests."""
    from kestrel_sovereign.voice.base import VoiceConfig, VoiceInfo
    from kestrel_sovereign.features.voice.feature import VoicePrivacyError

    vf = MagicMock()
    vf._voice_config = VoiceConfig(
        tts_provider="openai",
        tts_voice_id="nova",
        tts_model="tts-1",
        stt_provider="deepgram",
        output_format="opus",
    )
    vf._get_privacy_mode_name = MagicMock(return_value="normal")
    vf._cloud_allowed = MagicMock(return_value=cloud_allowed)
    vf._get_audio_storage_policy = MagicMock(return_value="full")

    # TTS provider mock
    tts = AsyncMock()
    tts.list_voices = AsyncMock(return_value=[
        VoiceInfo(voice_id="nova", name="Nova", provider="openai"),
    ])
    tts.synthesize = AsyncMock(return_value=b"\x00\x01\x02tts-audio")
    tts.name = "openai"

    async def _tts_stream(*a, **kw):
        yield b"audio-chunk-1"
        yield b"audio-chunk-2"

    tts.synthesize_stream = _tts_stream

    if has_tts:
        if privacy_error == "tts":
            vf._get_tts_provider = AsyncMock(
                side_effect=VoicePrivacyError("TTS blocked by privacy")
            )
        else:
            vf._get_tts_provider = AsyncMock(return_value=tts)
    else:
        vf._get_tts_provider = AsyncMock(
            side_effect=Exception("No TTS provider available.")
        )

    # STT provider mock
    stt = AsyncMock()
    stt.transcribe = AsyncMock(return_value="hello agent")
    stt.name = "deepgram"

    if stt_streaming:
        # Match the STTProvider ABC: transcribe_stream takes audio_format
        # and sample_rate kwargs added so the WS handler can pin Whisper to
        # raw PCM16 + a known rate.
        async def _stt_stream(audio_iter, language="", audio_format="pcm16", sample_rate=24000):
            async for chunk in audio_iter:
                yield "hello agent"

        stt.transcribe_stream = _stt_stream

    if has_stt:
        if privacy_error == "stt":
            vf._get_stt_provider = AsyncMock(
                side_effect=VoicePrivacyError("STT blocked by privacy")
            )
        else:
            vf._get_stt_provider = AsyncMock(return_value=stt)
    else:
        vf._get_stt_provider = AsyncMock(
            side_effect=Exception("No STT provider available.")
        )

    return vf


def _make_agent(voice_feature=None, has_streaming=True):
    """Build a mock agent with optional VoiceFeature."""
    agent = MagicMock()
    features = {}
    if voice_feature:
        features["VoiceFeature"] = voice_feature
    agent.features = features
    agent.identity = MagicMock()
    agent.identity.voice_config = None

    if has_streaming:
        async def _process_streaming(text, **kwargs):
            yield "Hello, "
            yield "how can I help?"

        agent.process_input_streaming = _process_streaming
    else:
        agent.process_input = AsyncMock(return_value="Hello, how can I help?")
        # Remove streaming attr so endpoint uses process_input
        if hasattr(agent, "process_input_streaming"):
            del agent.process_input_streaming

    return agent


def _prepare_app(agent):
    """Prepare FastAPI app with mock agent for testing."""
    from server import app
    from kestrel_sovereign.endpoints.voice import router as voice_router

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
    # Voice router is now feature-contributed; mount it for tests
    _existing_paths = {getattr(r, "path", "") for r in app.routes}
    if "/voice/voices" not in _existing_paths:
        app.include_router(voice_router)
    return app, original


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


# ------------------------------------------------------------------
# Binary framing tests
# ------------------------------------------------------------------


class TestBinaryFraming:
    """Test the binary framing encode/decode helpers."""

    def test_encode_control_json(self):
        payload = {"type": "status", "state": "listening"}
        encoded = encode_control(payload)
        assert encoded[0] == FRAME_JSON
        decoded = json.loads(encoded[1:])
        assert decoded == payload

    def test_encode_audio_frame(self):
        audio = b"\xff\xfe\x00\x01opus-data"
        encoded = encode_audio(audio)
        assert encoded[0] == FRAME_AUDIO
        assert encoded[1:] == audio

    def test_decode_control_frame(self):
        payload = {"type": "transcript", "text": "hello", "final": True}
        encoded = encode_control(payload)
        frame_type, decoded = _decode_frame(encoded)
        assert frame_type == FRAME_JSON
        assert decoded == payload

    def test_decode_audio_frame(self):
        audio = b"\x00\x01\x02\x03"
        encoded = encode_audio(audio)
        frame_type, decoded = _decode_frame(encoded)
        assert frame_type == FRAME_AUDIO
        assert decoded == audio

    def test_frame_type_byte_values(self):
        assert FRAME_JSON == 0x00
        assert FRAME_AUDIO == 0x01


# ------------------------------------------------------------------
# WebSocket connection tests
# ------------------------------------------------------------------


class TestVoiceChatConnection:
    """Test WebSocket connection establishment and rejection."""

    def test_connection_accepted_with_valid_auth(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    with client.websocket_connect(
                        "/voice/chat?api_key=test-key"
                    ) as ws:
                        # Should receive initial status: listening
                        data = ws.receive_bytes()
                        frame_type, payload = _decode_frame(data)
                        assert frame_type == FRAME_JSON
                        assert payload["type"] == "status"
                        assert payload["state"] == "listening"
        finally:
            _restore_app(app, orig)

    def test_connection_rejected_without_auth(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    # No api_key → should be rejected
                    with pytest.raises(Exception):
                        with client.websocket_connect("/voice/chat") as ws:
                            ws.receive_bytes()
        finally:
            _restore_app(app, orig)

    def test_connection_rejected_with_wrong_key(self):
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    with pytest.raises(Exception):
                        with client.websocket_connect(
                            "/voice/chat?api_key=wrong-key"
                        ) as ws:
                            ws.receive_bytes()
        finally:
            _restore_app(app, orig)

    def test_connection_rejected_when_no_agent(self):
        app, orig = _prepare_app(MagicMock())
        # Set agent to None
        app.state.agent = None
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    with pytest.raises(Exception):
                        with client.websocket_connect(
                            "/voice/chat?api_key=test-key"
                        ) as ws:
                            ws.receive_bytes()
        finally:
            _restore_app(app, orig)

    def test_connection_rejected_when_no_voice_feature(self):
        agent = _make_agent(voice_feature=None)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    with pytest.raises(Exception):
                        with client.websocket_connect(
                            "/voice/chat?api_key=test-key"
                        ) as ws:
                            ws.receive_bytes()
        finally:
            _restore_app(app, orig)

    def test_open_access_when_no_api_key_configured(self):
        """When KESTREL_API_KEY is empty, auth is open."""
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": ""}, clear=False):
                with TestClient(app) as client:
                    with client.websocket_connect("/voice/chat") as ws:
                        data = ws.receive_bytes()
                        frame_type, payload = _decode_frame(data)
                        assert payload["state"] == "listening"
        finally:
            _restore_app(app, orig)


# ------------------------------------------------------------------
# Privacy mode rejection
# ------------------------------------------------------------------


class TestVoiceChatPrivacy:
    """Test privacy mode enforcement at connection time."""

    def test_stt_privacy_rejection_closes_connection(self):
        vf = _make_voice_feature(privacy_error="stt")
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    with pytest.raises(Exception):
                        with client.websocket_connect(
                            "/voice/chat?api_key=test-key"
                        ) as ws:
                            ws.receive_bytes()
        finally:
            _restore_app(app, orig)

    def test_tts_privacy_rejection_closes_connection(self):
        vf = _make_voice_feature(privacy_error="tts")
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    with pytest.raises(Exception):
                        with client.websocket_connect(
                            "/voice/chat?api_key=test-key"
                        ) as ws:
                            ws.receive_bytes()
        finally:
            _restore_app(app, orig)

    def test_stt_unavailable_closes_connection(self):
        vf = _make_voice_feature(has_stt=False)
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    with pytest.raises(Exception):
                        with client.websocket_connect(
                            "/voice/chat?api_key=test-key"
                        ) as ws:
                            ws.receive_bytes()
        finally:
            _restore_app(app, orig)


# ------------------------------------------------------------------
# State machine transitions
# ------------------------------------------------------------------


class TestVoiceChatStateMachine:
    """Test state machine transitions: listening → thinking → speaking → listening."""

    def test_full_voice_loop_state_transitions(self, disable_vad):
        """Send audio, verify: listening → transcript → thinking → response → speaking → audio → listening."""
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    with client.websocket_connect(
                        "/voice/chat?api_key=test-key"
                    ) as ws:
                        # 1. Initial status: listening
                        data = ws.receive_bytes()
                        _, payload = _decode_frame(data)
                        assert payload == {"type": "status", "state": "listening"}

                        # 2. Send audio frame
                        ws.send_bytes(b"\x00\x01\x02fake-opus-audio")

                        # Collect all messages until we get back to listening
                        messages = []
                        for _ in range(20):  # Safety limit
                            try:
                                raw = ws.receive_bytes()
                                ft, msg = _decode_frame(raw)
                                messages.append((ft, msg))
                                # Stop when we return to listening state
                                if ft == FRAME_JSON and msg.get("type") == "status" and msg.get("state") == "listening":
                                    break
                            except Exception:
                                break

                        # Verify we got the expected state transitions
                        json_messages = [m for ft, m in messages if ft == FRAME_JSON]
                        audio_messages = [m for ft, m in messages if ft == FRAME_AUDIO]

                        # Should have transcript
                        transcripts = [m for m in json_messages if m.get("type") == "transcript"]
                        assert len(transcripts) >= 1
                        assert transcripts[0]["final"] is True

                        # Should have thinking status
                        statuses = [m for m in json_messages if m.get("type") == "status"]
                        status_states = [s["state"] for s in statuses]
                        assert "thinking" in status_states

                        # Should have response text
                        responses = [m for m in json_messages if m.get("type") == "response"]
                        assert len(responses) >= 1

                        # Should have speaking status
                        assert "speaking" in status_states

                        # Should have audio chunks
                        assert len(audio_messages) >= 1

                        # Should return to listening
                        assert status_states[-1] == "listening"
        finally:
            _restore_app(app, orig)

    def test_client_text_end_message_closes_connection(self):
        """Client sending {"type": "end"} should close gracefully."""
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    with client.websocket_connect(
                        "/voice/chat?api_key=test-key"
                    ) as ws:
                        # Receive initial listening status
                        ws.receive_bytes()
                        # Send end message
                        ws.send_text(json.dumps({"type": "end"}))
                        # Connection should close gracefully (no error)
        finally:
            _restore_app(app, orig)

    def test_invalid_json_from_client_returns_error(self):
        """Sending invalid JSON text should return an error control message."""
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    with client.websocket_connect(
                        "/voice/chat?api_key=test-key"
                    ) as ws:
                        # Receive initial listening status
                        ws.receive_bytes()
                        # Send invalid JSON
                        ws.send_text("not-valid-json{{{")
                        # Should get an error control message
                        data = ws.receive_bytes()
                        _, payload = _decode_frame(data)
                        assert payload["type"] == "error"
                        assert "Invalid JSON" in payload["message"]
        finally:
            _restore_app(app, orig)

    def test_agent_fallback_to_process_input(self, disable_vad):
        """When agent lacks process_input_streaming, fall back to process_input."""
        vf = _make_voice_feature()
        agent = _make_agent(vf, has_streaming=False)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    with client.websocket_connect(
                        "/voice/chat?api_key=test-key"
                    ) as ws:
                        # Initial listening
                        ws.receive_bytes()

                        # Send audio
                        ws.send_bytes(b"\x00\x01fake-audio")

                        # Collect messages
                        messages = []
                        for _ in range(20):
                            try:
                                raw = ws.receive_bytes()
                                ft, msg = _decode_frame(raw)
                                messages.append((ft, msg))
                                if ft == FRAME_JSON and msg.get("type") == "status" and msg.get("state") == "listening":
                                    break
                            except Exception:
                                break

                        json_messages = [m for ft, m in messages if ft == FRAME_JSON]
                        responses = [m for m in json_messages if m.get("type") == "response"]
                        assert len(responses) >= 1
                        assert "Hello" in responses[0]["text"]
        finally:
            _restore_app(app, orig)


# ------------------------------------------------------------------
# Disconnect handling
# ------------------------------------------------------------------


class TestVoiceChatDisconnect:
    """Test graceful disconnect handling."""

    def test_client_disconnect_handled_gracefully(self):
        """Client disconnecting should not raise an error."""
        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    with client.websocket_connect(
                        "/voice/chat?api_key=test-key"
                    ) as ws:
                        # Receive initial status
                        ws.receive_bytes()
                        # Close from client side — should not raise
                        ws.close()
        finally:
            _restore_app(app, orig)

    def test_vad_task_exits_on_client_disconnect(self):
        """VAD processing is request-scoped and awaited during disconnect cleanup."""
        started = threading.Event()
        cleaned_up = threading.Event()

        class FakeVAD:
            async def detect_utterances(self, stream):
                try:
                    async for _frame in stream:
                        started.set()
                        yield "speech_start", b""
                        await asyncio.Event().wait()
                finally:
                    cleaned_up.set()

        fake_vad_module = types.ModuleType("kestrel_sovereign.voice.vad")
        fake_vad_module.VoiceActivityDetector = lambda **_kwargs: FakeVAD()
        fake_vad_module.load_vad_config = lambda _config: {
            "aggressiveness": 1,
            "silence_threshold_ms": 100,
            "pre_speech_padding_ms": 20,
        }

        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        vad_key = "kestrel_sovereign.voice.vad"
        saved_vad = sys.modules.get(vad_key)

        try:
            sys.modules[vad_key] = fake_vad_module
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    with client.websocket_connect(
                        "/voice/chat?api_key=test-key"
                    ) as ws:
                        ws.receive_bytes()
                        ws.send_bytes(b"\x00\x01fake-audio")
                        assert started.wait(timeout=1)
                        ws.close()

            assert cleaned_up.wait(timeout=1)
        finally:
            if saved_vad is not None:
                sys.modules[vad_key] = saved_vad
            else:
                sys.modules.pop(vad_key, None)
            _restore_app(app, orig)


# ---------------------------------------------------------------------------
# WS-must-force-Pipeline regression
# ---------------------------------------------------------------------------


class TestWsForcesPipelineRoute:
    """The /voice/chat endpoint IS the Pipeline transport. The handler must
    pass `prefer_realtime=False` when asking for STT/TTS providers, otherwise
    the resolver picks Realtime (which legitimately has no STT) and the
    handler closes 4403 'No STT provider available' — the bug the user hit
    after merging the multi_agent WS routing fix.
    """

    def test_handler_passes_prefer_realtime_false_to_provider_getters(self):
        from kestrel_sovereign.voice.routing import UserVoicePreferences

        vf = _make_voice_feature()
        agent = _make_agent(vf)
        app, orig = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    with client.websocket_connect("/voice/chat?api_key=test-key") as ws:
                        ws.receive_bytes()  # initial status
                        ws.close()

            # Inspect what the handler called _get_stt_provider with.
            stt_calls = vf._get_stt_provider.await_args_list
            tts_calls = vf._get_tts_provider.await_args_list
            assert stt_calls, "WS handler never called _get_stt_provider"
            assert tts_calls, "WS handler never called _get_tts_provider"

            stt_overrides = stt_calls[0].kwargs.get("overrides")
            tts_overrides = tts_calls[0].kwargs.get("overrides")
            assert isinstance(stt_overrides, UserVoicePreferences), (
                "_get_stt_provider must be called with an overrides kwarg "
                "(got args=%r kwargs=%r)" % (stt_calls[0].args, stt_calls[0].kwargs)
            )
            assert stt_overrides.prefer_realtime is False, (
                "WS handler must force prefer_realtime=False; otherwise "
                "the resolver picks Realtime and there's no STT to return"
            )
            assert isinstance(tts_overrides, UserVoicePreferences)
            assert tts_overrides.prefer_realtime is False
        finally:
            _restore_app(app, orig)
