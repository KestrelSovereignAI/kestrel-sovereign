"""
Base classes for voice providers (TTS and STT).

Re-exports from kestrel_sdk.voice.base for backward compatibility.
Feature packages should import from kestrel_sdk.voice.base directly.
"""

# Re-export everything from kestrel_sdk
from kestrel_sdk.voice.base import (  # noqa: F401
    VoiceInfo,
    VoiceConfig,
    match_voice,
    split_sentences,
    TTSProvider,
    STTProvider,
)
