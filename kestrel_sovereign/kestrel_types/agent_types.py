"""
Agent Protocol definitions.

Defines interfaces for agents without importing concrete implementations,
breaking circular dependencies.
"""

from typing import Protocol, Any, Dict, List, Optional


class AgentProtocol(Protocol):
    """Protocol for Kestrel agents."""

    agent_id: str
    did: Optional[str]

    async def initialize(self) -> None:
        """Initialize the agent."""
        ...

    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a message and return a response."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the agent and release resources."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Get the current agent state."""
        ...
