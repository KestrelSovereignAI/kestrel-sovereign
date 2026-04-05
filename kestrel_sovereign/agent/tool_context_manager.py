"""
Tool Context Manager for Kestrel Agent.

Handles tool execution context, status reporting, and agent-accessible context management.
Extracted from ContextManager to improve modularity and maintainability.
"""

import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timezone

if TYPE_CHECKING:
    from kestrel_sovereign.storage import AsyncStorage

logger = logging.getLogger(__name__)


class ToolContextManager:
    """
    Manages tool execution context and agent-accessible context operations.

    Responsibilities:
    1. Provide context status for agent introspection
    2. Handle agent-accessible context management tools
    3. Manage context window utilization reporting
    4. Coordinate with other managers for comprehensive status
    """

    def __init__(
        self,
        storage: "AsyncStorage",
        model: str = "auto",
        agent_id: Optional[str] = None,
        llm_service=None,
    ):
        """
        Initialize the tool context manager.

        Args:
            storage: AsyncStorage instance for context operations
            model: Model name for token counting/limits
            agent_id: Agent ID for scoped queries
            llm_service: Optional LLMService for live model resolution
        """
        self.storage = storage
        self._model = model
        self.agent_id = agent_id
        self._llm_service = llm_service

    @property
    def model(self) -> str:
        """Resolve the current model dynamically when llm_service is available."""
        if self._llm_service:
            return self._llm_service.get_active_model_id()
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    async def get_status(self, counter, history: Optional[list] = None, session_id: Optional[str] = None, context_stats=None) -> Dict[str, Any]:
        """
        Get detailed context window status for agent introspection.

        Uses the same data path as the LLM to ensure the status report
        reflects what the model actually receives.

        Args:
            counter: TokenCounter for token counting
            history: Pre-fetched session-filtered history (preferred).
                     If None, falls back to limited fetch to approximate LLM path.
            session_id: Optional session ID for scoped history fetch.

        Returns:
            Comprehensive information about context utilization and budget.
        """
        from .token_budget import create_budget

        # Use provided history (same as LLM sees) or fetch with same constraints
        if history is None:
            conv_store = self._get_conversation_store()
            if conv_store:
                # Mirror the LLM path: limited to 50 messages
                all_history = await conv_store.get_full_history()
                history = all_history[-50:] if len(all_history) > 50 else all_history
            else:
                history = []

        message_count = len(history)
        budget = create_budget(self.model, message_count, adaptive=True)

        # Calculate tokens by category — this mirrors what format_conversation_history does
        history_tokens = sum(
            counter.count(m.get("content", ""))
            for m in history
        )

        # Get oldest message timestamp
        oldest_timestamp = None
        if history and history[0].get("created_at"):
            oldest_timestamp = history[0].get("created_at")

        # Report utilization per-allocation AND total
        history_utilization = (history_tokens / budget.history * 100) if budget.history > 0 else 0
        total_utilization = (history_tokens / budget.total_budget * 100) if budget.total_budget > 0 else 0

        status = {
            "success": True,
            "total_budget": budget.total_budget,
            "total_used": history_tokens,
            "utilization_percent": round(total_utilization, 1),
            "history_utilization_percent": round(history_utilization, 1),
            "compression_recommended": history_utilization >= 70.0,
            "allocations": {
                "system": {
                    "budget": budget.system,
                    "percent": round(budget.system / budget.total_budget * 100, 1) if budget.total_budget > 0 else 0
                },
                "history": {
                    "budget": budget.history,
                    "used": history_tokens,
                    "percent": round(history_utilization, 1),
                },
                "episodes": {
                    "budget": budget.episodes,
                    "percent": round(budget.episodes / budget.total_budget * 100, 1) if budget.total_budget > 0 else 0
                },
                "memories": {
                    "budget": budget.memories,
                    "percent": round(budget.memories / budget.total_budget * 100, 1) if budget.total_budget > 0 else 0
                },
                "rag": {
                    "budget": budget.rag,
                    "percent": round(budget.rag / budget.total_budget * 100, 1) if budget.total_budget > 0 else 0
                }
            },
            "message_count": message_count,
            "oldest_message_age": oldest_timestamp,
            "model": self.model,
            "note": "Status reflects the same data path used by the LLM (session-filtered, limited to 50 messages)."
        }

        # Nest context analysis from the accumulator (if provided)
        if context_stats is not None:
            status["analysis"] = context_stats.get_analysis()

        return status

    def get_budget_status(self, message_count: int = 0) -> Dict[str, Any]:
        """
        Get current token budget status.

        Useful for UI display or debugging.
        """
        from .token_budget import create_budget

        budget = create_budget(self.model, message_count, adaptive=True)
        return budget.get_summary()

    def _get_conversation_store(self):
        """Get the conversation store from storage hierarchy."""
        if hasattr(self.storage, 'conversation'):
            return self.storage.conversation
        elif hasattr(self.storage, '_storage') and hasattr(self.storage._storage, 'conversation'):
            return self.storage._storage.conversation
        return None
