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

    async def get_status(self, counter) -> Dict[str, Any]:
        """
        Get detailed context window status for agent introspection.

        Returns comprehensive information about context utilization,
        budget allocation, and recommendations.
        """
        from .token_budget import create_budget

        # Get conversation history
        history = []
        conv_store = self._get_conversation_store()
        if conv_store:
            history = await conv_store.get_full_history()

        message_count = len(history)
        budget = create_budget(self.model, message_count, adaptive=True)

        # Calculate tokens by category
        history_tokens = sum(
            counter.count(m.get("content", ""))
            for m in history
        )

        # Get oldest message timestamp
        oldest_timestamp = None
        if history and history[0].get("created_at"):
            oldest_timestamp = history[0].get("created_at")

        # Calculate utilization
        total_used = history_tokens  # Base from history
        utilization = (total_used / budget.total_budget * 100) if budget.total_budget > 0 else 0

        return {
            "success": True,
            "total_budget": budget.total_budget,
            "total_used": total_used,
            "utilization_percent": round(utilization, 1),
            "compression_recommended": utilization >= 70.0,
            "allocations": {
                "system": {
                    "budget": budget.system,
                    "percent": round(budget.system / budget.total_budget * 100, 1) if budget.total_budget > 0 else 0
                },
                "history": {
                    "budget": budget.history,
                    "used": history_tokens,
                    "percent": round(history_tokens / budget.history * 100, 1) if budget.history > 0 else 0
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
            "model": self.model
        }

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
