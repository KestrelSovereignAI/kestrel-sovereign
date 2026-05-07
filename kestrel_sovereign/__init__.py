"""
Kestrel Sovereign AI Agent Framework.

Constitutional AI with cryptographic identity and sovereign data ownership.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("kestrel-sovereign")
except PackageNotFoundError:
    __version__ = "0.0.0+local"
