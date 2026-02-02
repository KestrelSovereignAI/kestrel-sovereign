"""
Context Management Feature for Kestrel Agent.

Provides agent-accessible tools for managing the context window:
- Introspection (see current utilization)
- Selective summarization (compress specific sections)
- Content marking (protect/droppable)
- Proactive compression (agent-triggered)
- Soft removal (exclude from context)
"""

from .feature import ContextFeature

__all__ = ["ContextFeature"]
