"""
Kestrel Voice Provider Abstraction Layer.

Provides TTSProvider and STTProvider ABCs plus a VoiceProviderRegistry
for managing voice capabilities across local and cloud providers.

Core keeps Piper (local TTS) + FasterWhisper (local STT) + interfaces.
Cloud providers (ElevenLabs, Deepgram, OpenAI) are in separate packages
that register via the ``kestrel_sovereign.voice_providers`` entry_point group.
"""
from .base import TTSProvider, STTProvider, VoiceConfig, VoiceInfo
from .provider_registry import VoiceProviderRegistry

# Local providers — import errors are swallowed so the package works
# without optional voice-local dependencies installed.
try:
    from .piper_tts import PiperTTSProvider
except ImportError:
    PiperTTSProvider = None  # type: ignore[assignment,misc]

try:
    from .faster_whisper_stt import FasterWhisperSTTProvider
except ImportError:
    FasterWhisperSTTProvider = None  # type: ignore[assignment,misc]

__all__ = [
    "TTSProvider",
    "STTProvider",
    "VoiceConfig",
    "VoiceInfo",
    "VoiceProviderRegistry",
    "PiperTTSProvider",
    "FasterWhisperSTTProvider",
]
