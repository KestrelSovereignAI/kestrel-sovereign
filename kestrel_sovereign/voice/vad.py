"""Voice Activity Detection for WebSocket turn-taking.

Uses webrtcvad for reliable, low-latency voice activity detection with
a ring buffer approach for capturing pre-speech audio and configurable
silence thresholds to avoid splitting mid-sentence pauses.
"""

import collections
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_AGGRESSIVENESS = 2
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_DURATION_MS = 30
DEFAULT_SILENCE_THRESHOLD_MS = 800
DEFAULT_PRE_SPEECH_PADDING_MS = 300


class VoiceActivityDetector:
    """Detects speech start/stop in audio streams for turn-taking.

    Uses webrtcvad for reliable, low-latency voice activity detection.
    """

    def __init__(
        self,
        aggressiveness: int = DEFAULT_AGGRESSIVENESS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
        silence_threshold_ms: int = DEFAULT_SILENCE_THRESHOLD_MS,
        pre_speech_padding_ms: int = DEFAULT_PRE_SPEECH_PADDING_MS,
    ):
        """
        Args:
            aggressiveness: 0-3, higher = more aggressive filtering of non-speech.
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000).
            frame_duration_ms: Frame size (10, 20, or 30 ms).
            silence_threshold_ms: How long silence before "speech_end" fires.
            pre_speech_padding_ms: How much audio to keep before speech start.
        """
        import webrtcvad

        if aggressiveness not in range(4):
            raise ValueError(f"aggressiveness must be 0-3, got {aggressiveness}")
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(
                f"sample_rate must be 8000/16000/32000/48000, got {sample_rate}"
            )
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError(
                f"frame_duration_ms must be 10/20/30, got {frame_duration_ms}"
            )

        self._vad = webrtcvad.Vad(aggressiveness)
        self._sample_rate = sample_rate
        self._frame_duration_ms = frame_duration_ms
        self._silence_threshold_ms = silence_threshold_ms
        self._pre_speech_padding_ms = pre_speech_padding_ms

        # Compute derived values
        self._frame_byte_size = 2 * sample_rate * frame_duration_ms // 1000
        self._silence_frame_count = max(
            1, silence_threshold_ms // frame_duration_ms
        )
        self._ring_buffer_size = max(
            1, pre_speech_padding_ms // frame_duration_ms
        )

    @property
    def frame_byte_size(self) -> int:
        """Expected byte size for a single audio frame (16-bit PCM)."""
        return self._frame_byte_size

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_duration_ms(self) -> int:
        return self._frame_duration_ms

    def is_speech(self, audio_frame: bytes) -> bool:
        """Check if a single audio frame contains speech.

        Args:
            audio_frame: Raw 16-bit PCM audio, must be exactly
                frame_duration_ms long at the configured sample rate.

        Returns:
            True if the frame contains speech.
        """
        return self._vad.is_speech(audio_frame, self._sample_rate)

    async def detect_utterances(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[tuple[str, bytes]]:
        """Detect speech utterances in an audio stream.

        Yields:
            ("speech_start", b"") - user started speaking
            ("speech_data", audio_bytes) - audio data during speech
            ("speech_end", accumulated_audio) - user stopped speaking,
                includes all audio from the utterance (with pre-speech padding)

        Uses a ring buffer and silence counting to handle:
        - Pre-speech audio (capture onset via ring buffer)
        - Post-speech silence (don't cut off too early)
        - Short pauses within speech (don't split mid-sentence)
        """
        ring_buffer: collections.deque[bytes] = collections.deque(
            maxlen=self._ring_buffer_size
        )
        in_speech = False
        silence_count = 0
        speech_audio: list[bytes] = []

        async for frame in audio_stream:
            if len(frame) != self._frame_byte_size:
                # Skip malformed frames — client may send partial data
                logger.debug(
                    "Skipping frame: expected %d bytes, got %d",
                    self._frame_byte_size,
                    len(frame),
                )
                continue

            is_speech = self.is_speech(frame)

            if not in_speech:
                # Not currently in speech — watch for speech onset
                if is_speech:
                    in_speech = True
                    silence_count = 0

                    # Include ring buffer contents as pre-speech audio
                    speech_audio = list(ring_buffer)
                    speech_audio.append(frame)
                    ring_buffer.clear()

                    yield ("speech_start", b"")
                    # Yield the pre-speech + onset frames as speech_data
                    for buffered_frame in speech_audio:
                        yield ("speech_data", buffered_frame)
                else:
                    ring_buffer.append(frame)
            else:
                # Currently in speech
                speech_audio.append(frame)
                yield ("speech_data", frame)

                if not is_speech:
                    silence_count += 1
                    if silence_count >= self._silence_frame_count:
                        # Enough silence — end the utterance
                        accumulated = b"".join(speech_audio)
                        yield ("speech_end", accumulated)

                        # Reset state
                        in_speech = False
                        silence_count = 0
                        speech_audio = []
                else:
                    silence_count = 0

        # Stream ended while in speech — flush remaining audio
        if in_speech and speech_audio:
            accumulated = b"".join(speech_audio)
            yield ("speech_end", accumulated)


def load_vad_config(config: dict | None = None) -> dict:
    """Extract VAD configuration from a parsed kestrel.toml dict.

    Args:
        config: Parsed TOML dict (the full config). If None, returns defaults.

    Returns:
        Dict with keys: aggressiveness, silence_threshold_ms, pre_speech_padding_ms.
    """
    defaults = {
        "aggressiveness": DEFAULT_AGGRESSIVENESS,
        "silence_threshold_ms": DEFAULT_SILENCE_THRESHOLD_MS,
        "pre_speech_padding_ms": DEFAULT_PRE_SPEECH_PADDING_MS,
    }
    if config is None:
        return defaults

    vad_section = config.get("voice", {}).get("vad", {})
    return {
        "aggressiveness": vad_section.get("aggressiveness", defaults["aggressiveness"]),
        "silence_threshold_ms": vad_section.get(
            "silence_threshold_ms", defaults["silence_threshold_ms"]
        ),
        "pre_speech_padding_ms": vad_section.get(
            "pre_speech_padding_ms", defaults["pre_speech_padding_ms"]
        ),
    }
