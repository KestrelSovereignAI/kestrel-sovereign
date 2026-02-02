"""
Feature and Tool Protocol definitions.

Defines interfaces for features and tools without importing concrete
implementations, breaking circular dependencies.
"""

from typing import Protocol, Any, Dict, List, Optional


class ToolProtocol(Protocol):
    """Protocol for agent tools."""

    name: str
    description: str

    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        ...


class FeatureProtocol(Protocol):
    """Protocol for agent features."""

    name: str
    enabled: bool

    async def initialize(self) -> None:
        """Initialize the feature."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the feature and release resources."""
        ...

    def get_tools(self) -> List[ToolProtocol]:
        """Get the list of tools provided by this feature."""
        ...

    async def handle_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Handle a message and optionally return a response."""
        ...
