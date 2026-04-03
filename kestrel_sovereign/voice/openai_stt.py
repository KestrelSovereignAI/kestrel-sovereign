"""
OpenAI STT Provider — re-export from extracted package.

The implementation has been moved to kestrel_voice_openai.
This module provides backward-compatible imports.
"""
from kestrel_voice_openai.openai_stt import (  # noqa: F401
    OpenAISTTProvider,
    MAX_FILE_SIZE,
    SUPPORTED_FORMATS,
)
