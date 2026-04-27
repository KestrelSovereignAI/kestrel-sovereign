"""Sandbox backends for the computer-use feature."""

from .base import (
    CapabilityBlocked,
    CompletedRun,
    DirEntry,
    SandboxBackend,
)
from .docker import DockerSandboxBackend
from .local import LocalSandboxBackend

__all__ = [
    "CapabilityBlocked",
    "CompletedRun",
    "DirEntry",
    "DockerSandboxBackend",
    "LocalSandboxBackend",
    "SandboxBackend",
]
