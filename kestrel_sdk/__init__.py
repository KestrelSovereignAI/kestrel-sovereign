"""
Kestrel Sovereign SDK — lightweight interfaces for feature packages.

This package contains only the abstract base classes, protocols, and data models
that feature packages need to develop against. It has minimal dependencies
(pydantic only) so feature developers don't need to install the full
kestrel-sovereign framework.

Usage:
    from kestrel_sdk.features.base import Feature, tool
    from kestrel_sdk.hooks.base import Hook, HookEvent
    from kestrel_sdk.tools.base import AgentTool, ToolSchema
"""

__version__ = "0.1.0"
