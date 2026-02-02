"""
Agent module for Kestrel.

This module provides modular components for the Kestrel Agent:
- ContextManager: Unified context orchestration (primary interface)
- ContextBuilder: Assembles context for LLM prompts
- ConstitutionMixin: Constitution verification methods
- StreamingMixin: Streaming response methods
- BackupMixin: Backup/restore methods
- TokenCounter: Token counting with tiktoken/fallback
- TokenBudget: Model-aware token budget allocation
"""

from .context_builder import ContextBuilder
from .context_manager import ContextManager, ContextResult
from .constitution import ConstitutionMixin
from .streaming import StreamingMixin
from .backup import BackupMixin
from .token_counter import TokenCounter, get_token_counter
from .token_budget import TokenBudget, AdaptiveTokenBudget, create_budget

__all__ = [
    'ContextManager',
    'ContextResult',
    'ContextBuilder',
    'ConstitutionMixin',
    'StreamingMixin',
    'BackupMixin',
    'TokenCounter',
    'get_token_counter',
    'TokenBudget',
    'AdaptiveTokenBudget',
    'create_budget',
]
