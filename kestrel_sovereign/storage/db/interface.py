"""
Database Backend Interface

Abstract interface for database operations that works with both
SQLite (aiosqlite) and PostgreSQL (asyncpg).
"""

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, List, Optional, Sequence, Tuple, Union

# Type aliases
Params = Sequence[Any]
Row = Tuple[Any, ...]


class DatabaseBackend(ABC):
    """
    Abstract database backend - unified interface for SQLite and PostgreSQL.
    
    All queries use SQLite-style ? placeholders. The backend converts
    to $1, $2 style for PostgreSQL automatically.
    """
    
    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Return 'sqlite' or 'postgres'."""
        ...
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if database connection is active."""
        ...
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection."""
        ...
    
    @abstractmethod
    async def close(self) -> None:
        """Close database connection."""
        ...
    
    @abstractmethod
    async def execute(self, query: str, params: Params = ()) -> int:
        """
        Execute a write query (INSERT, UPDATE, DELETE).
        
        Args:
            query: SQL query with ? placeholders
            params: Query parameters
            
        Returns:
            Number of rows affected
        """
        ...
    
    @abstractmethod
    async def execute_many(self, query: str, params_list: List[Params]) -> int:
        """
        Execute query with multiple parameter sets.
        
        Args:
            query: SQL query with ? placeholders
            params_list: List of parameter tuples
            
        Returns:
            Total rows affected
        """
        ...
    
    @abstractmethod
    async def fetch_one(self, query: str, params: Params = ()) -> Optional[Row]:
        """
        Fetch a single row.
        
        Args:
            query: SQL query with ? placeholders
            params: Query parameters
            
        Returns:
            Row as tuple, or None if no results
        """
        ...
    
    @abstractmethod
    async def fetch_all(self, query: str, params: Params = ()) -> List[Row]:
        """
        Fetch all rows.
        
        Args:
            query: SQL query with ? placeholders
            params: Query parameters
            
        Returns:
            List of rows as tuples
        """
        ...
    
    @abstractmethod
    async def fetch_val(self, query: str, params: Params = ()) -> Optional[Any]:
        """
        Fetch a single value (first column of first row).
        
        Args:
            query: SQL query with ? placeholders
            params: Query parameters
            
        Returns:
            Single value, or None if no results
        """
        ...
    
    @abstractmethod
    async def execute_script(self, script: str) -> None:
        """
        Execute a multi-statement SQL script.
        
        Used for schema migrations and initialization.
        
        Args:
            script: SQL script (may contain multiple statements)
        """
        ...
    
    @asynccontextmanager
    @abstractmethod
    async def transaction(self) -> AsyncIterator[None]:
        """
        Transaction context manager.
        
        Usage:
            async with backend.transaction():
                await backend.execute("INSERT ...")
                await backend.execute("UPDATE ...")
        
        Commits on success, rolls back on exception.
        """
        ...
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        # Default implementation - backends may override for efficiency
        if self.backend_type == "sqlite":
            row = await self.fetch_one(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
        else:
            row = await self.fetch_one(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name=?",
                (table_name,)
            )
        return row is not None


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionError(DatabaseError):
    """Failed to connect to database."""
    pass


class QueryError(DatabaseError):
    """Query execution failed."""
    pass


class TransactionError(DatabaseError):
    """Transaction operation failed."""
    pass
