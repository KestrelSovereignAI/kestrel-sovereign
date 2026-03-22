"""
Agent module for Kestrel.

This module provides modular components for the Kestrel Agent:
- ContextManager: Unified context orchestration (primary interface)
- ContextBuilder: Assembles context for LLM prompts
- ConstitutionMixin: Constitution verification methods
- StreamingMixin: Streaming response methods
- BackupMixin: Backup/restore methods
- OrchestratorEngineMixin: Tool execution and orchestrator loop
- ToolRegistryMixin: Dynamic tool loading with LRU eviction
- ModelPreferenceMixin: Model selection and solvency
- EventManagerMixin: SSE events and notifications
- RequestLifecycleMixin: Request tracking and cancellation
- TokenCounter: Token counting with tiktoken/fallback
- TokenBudget: Model-aware token budget allocation
"""

from .context_builder import ContextBuilder
from .context_manager import ContextManager, ContextResult
from .constitution import ConstitutionMixin
from .streaming import StreamingMixin
from .backup import BackupMixin
from .orchestrator_engine import OrchestratorEngineMixin
from .tool_registry import ToolRegistryMixin
from .model_preference import ModelPreferenceMixin
from .event_manager import EventManagerMixin
from .request_lifecycle import RequestLifecycleMixin
from .token_counter import TokenCounter, get_token_counter
from .token_budget import TokenBudget, AdaptiveTokenBudget, create_budget

__all__ = [
    'ContextManager',
    'ContextResult',
    'ContextBuilder',
    'ConstitutionMixin',
    'StreamingMixin',
    'BackupMixin',
    'OrchestratorEngineMixin',
    'ToolRegistryMixin',
    'ModelPreferenceMixin',
    'EventManagerMixin',
    'RequestLifecycleMixin',
    'TokenCounter',
    'get_token_counter',
    'TokenBudget',
    'AdaptiveTokenBudget',
    'create_budget',
]
