"""
Type definitions and protocols for Kestrel.

This module contains Protocol-based interfaces to break circular dependencies
between core modules (storage, features, llm, security, a2a).

Instead of importing concrete implementations, modules can import these
lightweight Protocol definitions.
"""

from .storage_types import StorageProvider, DatabaseProvider
from .feature_types import FeatureProtocol, ToolProtocol
from .agent_types import AgentProtocol
from .llm_types import LLMRequest, LLMResponse

__all__ = [
    "StorageProvider",
    "DatabaseProvider",
    "FeatureProtocol",
    "ToolProtocol",
    "AgentProtocol",
    "LLMRequest",
    "LLMResponse",
]
