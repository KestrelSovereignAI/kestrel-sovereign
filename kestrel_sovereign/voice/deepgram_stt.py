"""
Deepgram STT Provider — re-export from extracted package.

The implementation has been moved to kestrel_voice_deepgram.
This module provides backward-compatible imports.
"""
try:
    from kestrel_voice_deepgram.deepgram_stt import (  # noqa: F401
        DeepgramSTTProvider,
        DEEPGRAM_AVAILABLE,
    )
except ImportError:
    DEEPGRAM_AVAILABLE = False
    DeepgramSTTProvider = None  # type: ignore[assignment,misc]
