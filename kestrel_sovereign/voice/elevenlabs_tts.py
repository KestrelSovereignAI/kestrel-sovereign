"""
ElevenLabs TTS Provider — re-export from extracted package.

The implementation has been moved to kestrel_voice_elevenlabs.
This module provides backward-compatible imports.
"""
from kestrel_voice_elevenlabs.elevenlabs_tts import (  # noqa: F401
    ElevenLabsTTSProvider,
    _FORMAT_MAP,
    _PCM_SAMPLE_RATES,
)
