"""
Kestrel Services.

High-level service classes that coordinate between features,
storage, and external providers.
"""

from .key_resolution import (
    KeyResolutionService,
    KeyNotConfiguredError,
    resolve_key,
)

__all__ = [
    "KeyResolutionService",
    "KeyNotConfiguredError",
    "resolve_key",
]
