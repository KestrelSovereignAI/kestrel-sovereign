"""Unit tests for Voice Activity Detection (kestrel_sovereign/voice/vad.py)."""

import asyncio
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock webrtcvad at module level so imports work without the real package.
# We track whether webrtcvad was already installed so we can clean up after.
# ---------------------------------------------------------------------------
_webrtcvad_was_installed = "webrtcvad" in sys.modules
_mock_webrtcvad = types.ModuleType("webrtcvad")
_mock_vad_class = MagicMock()
_mock_webrtcvad.Vad = _mock_vad_class
if not _webrtcvad_was_installed:
    sys.modules["webrtcvad"] = _mock_webrtcvad


@pytest.fixture(autouse=True, scope="module")
def _cleanup_webrtcvad_mock():
    """Remove the webrtcvad mock from sys.modules after all tests in this module."""
    yield
    if not _webrtcvad_was_installed:
        sys.modules.pop("webrtcvad", None)
        # Also remove cached vad module so it reimports cleanly
        sys.modules.pop("kestrel_sovereign.voice.vad", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pcm_frame(sample_rate: int = 16000, duration_ms: int = 30,
                    value: int = 0) -> bytes:
    """Create a silent PCM frame of the correct size (16-bit samples)."""
    num_samples = sample_rate * duration_ms // 1000
    return (value.to_bytes(2, "little", signed=True)) * num_samples


def _frame_byte_size(sample_rate: int = 16000, duration_ms: int = 30) -> int:
    return 2 * sample_rate * duration_ms // 1000


async def _async_iter(frames):
    """Turn a list into an async iterator."""
    for f in frames:
        yield f


def _fresh_vad(**kwargs):
    """Create a VoiceActivityDetector with a fresh mock Vad instance."""
    from kestrel_sovereign.voice.vad import VoiceActivityDetector

    mock_instance = MagicMock()
    _mock_vad_class.reset_mock()
    _mock_vad_class.return_value = mock_instance
    vad = VoiceActivityDetector(**kwargs)
    return vad, mock_instance


# ---------------------------------------------------------------------------
# Construction / validation tests
# ---------------------------------------------------------------------------


class TestVADConstruction:

    def test_default_construction(self):
        vad, mock_inst = _fresh_vad()
        _mock_vad_class.assert_called_once_with(2)
        assert vad.sample_rate == 16000
        assert vad.frame_duration_ms == 30
        assert vad.frame_byte_size == _frame_byte_size(16000, 30)

    def test_custom_parameters(self):
        vad, mock_inst = _fresh_vad(
            aggressiveness=3,
            sample_rate=8000,
            frame_duration_ms=20,
            silence_threshold_ms=600,
            pre_speech_padding_ms=200,
        )
        _mock_vad_class.assert_called_once_with(3)
        assert vad.sample_rate == 8000
        assert vad.frame_duration_ms == 20
        assert vad.frame_byte_size == _frame_byte_size(8000, 20)

    def test_invalid_aggressiveness_raises(self):
        from kestrel_sovereign.voice.vad import VoiceActivityDetector
        with pytest.raises(ValueError, match="aggressiveness"):
            VoiceActivityDetector(aggressiveness=5)

    def test_invalid_sample_rate_raises(self):
        from kestrel_sovereign.voice.vad import VoiceActivityDetector
        with pytest.raises(ValueError, match="sample_rate"):
            VoiceActivityDetector(sample_rate=44100)

    def test_invalid_frame_duration_raises(self):
        from kestrel_sovereign.voice.vad import VoiceActivityDetector
        with pytest.raises(ValueError, match="frame_duration_ms"):
            VoiceActivityDetector(frame_duration_ms=25)


# ---------------------------------------------------------------------------
# is_speech tests
# ---------------------------------------------------------------------------


class TestIsSpeech:

    def test_is_speech_returns_true_for_speech(self):
        vad, mock_inst = _fresh_vad()
        mock_inst.is_speech.return_value = True
        frame = _make_pcm_frame()
        assert vad.is_speech(frame) is True
        mock_inst.is_speech.assert_called_once_with(frame, 16000)

    def test_is_speech_returns_false_for_silence(self):
        vad, mock_inst = _fresh_vad()
        mock_inst.is_speech.return_value = False
        frame = _make_pcm_frame()
        assert vad.is_speech(frame) is False


# ---------------------------------------------------------------------------
# detect_utterances tests
# ---------------------------------------------------------------------------


class TestDetectUtterances:

    @pytest.mark.asyncio
    async def test_speech_start_and_end_events(self):
        """Verify speech_start followed by speech_end after silence threshold."""
        vad, mock_inst = _fresh_vad(
            silence_threshold_ms=90,  # 3 frames at 30ms
            pre_speech_padding_ms=30,  # 1 frame ring buffer
        )

        frame = _make_pcm_frame()
        # Sequence: 2 silence, 3 speech, 3 silence (triggers end)
        speech_pattern = [False, False, True, True, True, False, False, False]
        mock_inst.is_speech.side_effect = speech_pattern

        frames = [frame] * len(speech_pattern)
        events = []
        async for event_type, data in vad.detect_utterances(_async_iter(frames)):
            events.append((event_type, data))

        event_types = [e[0] for e in events]
        assert "speech_start" in event_types
        assert "speech_end" in event_types
        assert event_types.index("speech_start") < event_types.index("speech_end")

    @pytest.mark.asyncio
    async def test_ring_buffer_captures_pre_speech(self):
        """Ring buffer should include frames before speech onset."""
        vad, mock_inst = _fresh_vad(
            silence_threshold_ms=30,   # 1 frame triggers end
            pre_speech_padding_ms=60,  # 2-frame ring buffer
        )

        frame_size = vad.frame_byte_size
        # Create distinct frames so we can identify them
        silence1 = b"\x00\x00" * (frame_size // 2)
        silence2 = b"\x01\x00" * (frame_size // 2)
        speech1 = b"\x02\x00" * (frame_size // 2)
        silence3 = b"\x03\x00" * (frame_size // 2)

        mock_inst.is_speech.side_effect = [False, False, True, False]

        events = []
        async for event_type, data in vad.detect_utterances(
            _async_iter([silence1, silence2, speech1, silence3])
        ):
            events.append((event_type, data))

        event_types = [e[0] for e in events]
        assert "speech_start" in event_types
        assert "speech_data" in event_types

        # The speech_end accumulated audio should contain ring buffer + speech + silence
        speech_end_events = [e for e in events if e[0] == "speech_end"]
        assert len(speech_end_events) == 1
        accumulated = speech_end_events[0][1]
        assert silence1 in accumulated
        assert silence2 in accumulated
        assert speech1 in accumulated

    @pytest.mark.asyncio
    async def test_silence_threshold_prevents_premature_end(self):
        """Short silence during speech should not trigger speech_end."""
        vad, mock_inst = _fresh_vad(
            silence_threshold_ms=90,  # 3 frames needed
            pre_speech_padding_ms=30,
        )

        frame = _make_pcm_frame()
        # Speech, then 2 silence (< threshold of 3), then speech again, then 3 silence
        pattern = [True, True, False, False, True, True, False, False, False]
        mock_inst.is_speech.side_effect = pattern

        events = []
        async for event_type, data in vad.detect_utterances(
            _async_iter([frame] * len(pattern))
        ):
            events.append((event_type, data))

        speech_end_count = sum(1 for e in events if e[0] == "speech_end")
        assert speech_end_count == 1

    @pytest.mark.asyncio
    async def test_stream_end_flushes_in_progress_speech(self):
        """If stream ends during speech, remaining audio should be flushed."""
        vad, mock_inst = _fresh_vad(
            silence_threshold_ms=300,
            pre_speech_padding_ms=30,
        )

        frame = _make_pcm_frame()
        mock_inst.is_speech.side_effect = [True, True, True]

        events = []
        async for event_type, data in vad.detect_utterances(
            _async_iter([frame, frame, frame])
        ):
            events.append((event_type, data))

        event_types = [e[0] for e in events]
        assert "speech_start" in event_types
        assert "speech_end" in event_types

    @pytest.mark.asyncio
    async def test_malformed_frames_skipped(self):
        """Frames with wrong byte size should be silently skipped."""
        vad, mock_inst = _fresh_vad()

        events = []
        async for event_type, data in vad.detect_utterances(
            _async_iter([b"\x00\x01", b"\x00\x01\x02"])
        ):
            events.append((event_type, data))

        assert len(events) == 0
        mock_inst.is_speech.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_silence_no_events(self):
        """A stream of only silence should produce no speech events."""
        vad, mock_inst = _fresh_vad()
        mock_inst.is_speech.return_value = False

        frame = _make_pcm_frame()
        events = []
        async for event_type, data in vad.detect_utterances(
            _async_iter([frame] * 10)
        ):
            events.append((event_type, data))

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_multiple_utterances(self):
        """Multiple speech segments separated by sufficient silence."""
        vad, mock_inst = _fresh_vad(
            silence_threshold_ms=60,   # 2 frames
            pre_speech_padding_ms=30,  # 1 frame
        )

        frame = _make_pcm_frame()
        # Utterance 1: speech + silence, Utterance 2: speech + silence
        pattern = [True, True, False, False, True, True, False, False]
        mock_inst.is_speech.side_effect = pattern

        events = []
        async for event_type, data in vad.detect_utterances(
            _async_iter([frame] * len(pattern))
        ):
            events.append((event_type, data))

        speech_starts = sum(1 for e in events if e[0] == "speech_start")
        speech_ends = sum(1 for e in events if e[0] == "speech_end")
        assert speech_starts == 2
        assert speech_ends == 2


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------


class TestLoadVADConfig:

    def test_defaults_when_no_config(self):
        from kestrel_sovereign.voice.vad import load_vad_config
        cfg = load_vad_config(None)
        assert cfg["aggressiveness"] == 2
        assert cfg["silence_threshold_ms"] == 800
        assert cfg["pre_speech_padding_ms"] == 300

    def test_defaults_when_empty_config(self):
        from kestrel_sovereign.voice.vad import load_vad_config
        cfg = load_vad_config({})
        assert cfg["aggressiveness"] == 2
        assert cfg["silence_threshold_ms"] == 800
        assert cfg["pre_speech_padding_ms"] == 300

    def test_config_from_toml_dict(self):
        from kestrel_sovereign.voice.vad import load_vad_config
        toml_dict = {
            "voice": {
                "vad": {
                    "aggressiveness": 3,
                    "silence_threshold_ms": 500,
                    "pre_speech_padding_ms": 200,
                }
            }
        }
        cfg = load_vad_config(toml_dict)
        assert cfg["aggressiveness"] == 3
        assert cfg["silence_threshold_ms"] == 500
        assert cfg["pre_speech_padding_ms"] == 200

    def test_partial_config_uses_defaults_for_missing(self):
        from kestrel_sovereign.voice.vad import load_vad_config
        toml_dict = {
            "voice": {
                "vad": {
                    "aggressiveness": 1,
                }
            }
        }
        cfg = load_vad_config(toml_dict)
        assert cfg["aggressiveness"] == 1
        assert cfg["silence_threshold_ms"] == 800
        assert cfg["pre_speech_padding_ms"] == 300

    def test_config_without_vad_section(self):
        from kestrel_sovereign.voice.vad import load_vad_config
        toml_dict = {
            "voice": {
                "tts_provider_priority": ["piper"],
            }
        }
        cfg = load_vad_config(toml_dict)
        assert cfg["aggressiveness"] == 2
        assert cfg["silence_threshold_ms"] == 800
        assert cfg["pre_speech_padding_ms"] == 300
