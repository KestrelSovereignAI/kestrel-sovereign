"""
Kestrel Voice OpenAI — Cloud TTS + STT via OpenAI API.

Extracted from kestrel-sovereign as a standalone voice provider package.
Registers via entry_points group ``kestrel_sovereign.voice_providers``.
"""

from .openai_tts import OpenAITTSProvider
from .openai_stt import OpenAISTTProvider

__all__ = ["OpenAITTSProvider", "OpenAISTTProvider"]
