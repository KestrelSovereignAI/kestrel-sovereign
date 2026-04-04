"""
Base classes and interfaces for Kestrel agent tools.

This module re-exports from kestrel_sdk.tools.base for backward compatibility.
Feature packages should import from kestrel_sdk.tools.base directly.
"""

# Re-export everything from kestrel_sdk
from kestrel_sdk.tools.base import (  # noqa: F401
    AgentTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    ToolExecutionError,
)
