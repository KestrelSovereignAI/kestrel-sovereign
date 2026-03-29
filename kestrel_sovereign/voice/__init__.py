"""
Kestrel Voice Provider Abstraction Layer.

Provides TTSProvider and STTProvider ABCs plus a VoiceProviderRegistry
for managing voice capabilities across local and cloud providers.
"""
from .base import TTSProvider, STTProvider, VoiceConfig, VoiceInfo
from .openai_tts import OpenAITTSProvider
from .openai_stt import OpenAISTTProvider
from .piper_tts import PiperTTSProvider
from .provider_registry import VoiceProviderRegistry
from .elevenlabs_tts import ElevenLabsTTSProvider
from .faster_whisper_stt import FasterWhisperSTTProvider

# Optional providers — import errors are swallowed so the package works
# without optional dependencies installed.
try:
    from .deepgram_stt import DeepgramSTTProvider
except ImportError:
    DeepgramSTTProvider = None  # type: ignore[assignment,misc]

__all__ = [
    "TTSProvider",
    "STTProvider",
    "VoiceConfig",
    "VoiceInfo",
    "VoiceProviderRegistry",
    "OpenAITTSProvider",
    "OpenAISTTProvider",
    "PiperTTSProvider",
    "ElevenLabsTTSProvider",
    "FasterWhisperSTTProvider",
    "DeepgramSTTProvider",
]
