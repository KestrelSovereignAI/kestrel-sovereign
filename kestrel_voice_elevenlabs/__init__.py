"""
Kestrel Voice ElevenLabs — Cloud TTS via ElevenLabs API.

Extracted from kestrel-sovereign as a standalone voice provider package.
Registers via entry_points group ``kestrel_sovereign.voice_providers``.
"""

from .elevenlabs_tts import ElevenLabsTTSProvider

__all__ = ["ElevenLabsTTSProvider"]
