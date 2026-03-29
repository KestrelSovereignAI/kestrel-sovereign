"""
Kestrel Voice Provider Abstraction Layer.

Provides TTSProvider and STTProvider ABCs plus a VoiceProviderRegistry
for managing voice capabilities across local and cloud providers.
"""
from .base import TTSProvider, STTProvider, VoiceConfig, VoiceInfo
from .provider_registry import VoiceProviderRegistry

__all__ = [
    "TTSProvider",
    "STTProvider",
    "VoiceConfig",
    "VoiceInfo",
    "VoiceProviderRegistry",
]
