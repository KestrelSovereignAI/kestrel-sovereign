"""
Storage Protocol definitions.

Defines interfaces for storage and database providers without
importing concrete implementations, breaking circular dependencies.
"""

from typing import Protocol, Any, Optional, List, Dict
from datetime import datetime


class DatabaseProvider(Protocol):
    """Protocol for database connection providers."""

    async def execute(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> Any:
        """Execute a SQL query."""
        ...

    async def fetchone(self, query: str, params: Optional[tuple] = None) -> Optional[tuple]:
        """Fetch a single row."""
        ...

    async def fetchall(self, query: str, params: Optional[tuple] = None) -> List[tuple]:
        """Fetch all rows."""
        ...

    async def commit(self) -> None:
        """Commit the current transaction."""
        ...

    async def close(self) -> None:
        """Close the database connection."""
        ...


class StorageProvider(Protocol):
    """Protocol for general storage providers."""

    async def store(self, key: str, value: Any) -> None:
        """Store a value by key."""
        ...

    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        ...

    async def delete(self, key: str) -> None:
        """Delete a value by key."""
        ...

    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix."""
        ...

    async def close(self) -> None:
        """Close storage connections."""
        ...


class ConversationStore(Protocol):
    """Protocol for conversation storage."""

    async def add_message(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        **metadata
    ) -> str:
        """Add a message to the conversation history."""
        ...

    async def get_history(
        self,
        limit: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history."""
        ...

    async def clear_history(self) -> None:
        """Clear all conversation history."""
        ...
