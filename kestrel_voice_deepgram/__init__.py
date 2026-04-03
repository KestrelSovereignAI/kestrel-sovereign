"""
Kestrel Voice Deepgram — Cloud STT via Deepgram API.

Extracted from kestrel-sovereign as a standalone voice provider package.
Registers via entry_points group ``kestrel_sovereign.voice_providers``.
"""

from .deepgram_stt import DeepgramSTTProvider

__all__ = ["DeepgramSTTProvider"]
