"""
Database Backend Factory

Provides unified access to database backends based on configuration.

DATABASE BACKEND MODES:
    - SQLiteBackend (DEFAULT): Recommended for sovereign agents. Your data is a
      file you own. Supports offline-first operation, full portability, and the
      "two-file sovereign agent" vision (agent.db + emma.llamafile).

    - PostgresBackend (ADVANCED): For multi-tenant, high-concurrency, or server
      deployments. Use when you need concurrent writers, centralized analytics,
      or are running a managed service with multiple agents.

    Constitutional Council Decision (session 9282ed19):
        The council approved SQLite as the DEFAULT while retaining PostgreSQL for
        advanced use cases. See kestrel_sovereign/data/council_sessions/ for the
        full deliberation transcript.

    For cloud backup with SQLite: Use SyncService with S3Target or LighthouseTarget
    (see kestrel_sovereign/storage/sync/).
"""

import os
from typing import Any, Dict, Optional

from .interface import DatabaseBackend, DatabaseError, ConnectionError, QueryError, TransactionError
from .placeholder import sqlite_to_postgres, postgres_to_sqlite, normalize_schema
from .sqlite import SQLiteBackend

# Export all public interfaces
__all__ = [
    # Interface
    "DatabaseBackend",
    "DatabaseError",
    "ConnectionError",
    "QueryError",
    "TransactionError",
    # Backends
    "SQLiteBackend",
    "postgres_backend",  # PEP8 compliant factory function
    "PostgresBackend",  # Backwards compatibility alias (advanced mode)
    # Factory
    "get_backend",
    "create_backend",
    # Utilities
    "sqlite_to_postgres",
    "postgres_to_sqlite",
    "normalize_schema",
]


def postgres_backend(*args, **kwargs):
    """
    Factory function to create PostgresBackend instances.

    ADVANCED MODE: PostgreSQL is for multi-tenant, high-concurrency, or server
    deployments. For sovereign agents, SQLiteBackend is the recommended default.

    Use cases for PostgresBackend:
        - Multi-tenant SaaS with many agents sharing one database
        - High-concurrency workloads requiring concurrent writers
        - Centralized analytics and admin dashboards
        - Server deployments with connection pooling

    Lazy imports asyncpg to avoid requiring it when not needed.
    This is a factory function following PEP8 naming (snake_case).

    Args:
        *args: Passed to PostgresBackend constructor
        **kwargs: Passed to PostgresBackend constructor (dsn, host, etc.)

    Returns:
        PostgresBackend: A new PostgresBackend instance
    """
    from .postgres import PostgresBackend as _PostgresBackend
    return _PostgresBackend(*args, **kwargs)


# Backwards compatibility alias (advanced mode, use postgres_backend instead)
PostgresBackend = postgres_backend


async def get_backend(
    config: Optional[Dict[str, Any]] = None,
) -> DatabaseBackend:
    """
    Create and connect a database backend from configuration.
    
    Configuration options:
        backend: 'sqlite' or 'postgres' (default: 'sqlite')
        
        For SQLite:
            db_path: Path to database file (default: './data/kestrel.db')
        
        For PostgreSQL:
            dsn: Connection string (postgresql://user:pass@host:port/db)
            OR individual parameters:
            host: Database host
            port: Database port (default: 5432)
            database: Database name
            user: Database user
            password: Database password
            min_pool_size: Minimum pool connections (default: 2)
            max_pool_size: Maximum pool connections (default: 10)
    
    Environment variables (used if config not provided):
        KESTREL_DB_BACKEND: 'sqlite' or 'postgres'
        KESTREL_DB_PATH: SQLite database path
        KESTREL_DATABASE_URL: PostgreSQL connection string
        KESTREL_DB_HOST, KESTREL_DB_PORT, KESTREL_DB_NAME,
        KESTREL_DB_USER, KESTREL_DB_PASSWORD: PostgreSQL parameters
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Connected DatabaseBackend instance
    """
    backend = create_backend(config)
    await backend.connect()
    return backend


def create_backend(
    config: Optional[Dict[str, Any]] = None,
) -> DatabaseBackend:
    """
    Create a database backend from configuration (without connecting).
    
    See get_backend() for configuration options.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DatabaseBackend instance (not yet connected)
    """
    if config is None:
        config = {}
    
    # Get backend type from config or environment
    backend_type = config.get(
        "backend",
        os.getenv("KESTREL_DB_BACKEND", "sqlite")
    ).lower()
    
    if backend_type == "sqlite":
        return _create_sqlite_backend(config)
    elif backend_type in ("postgres", "postgresql"):
        return _create_postgres_backend(config)
    else:
        raise ValueError(f"Unknown database backend: {backend_type}")


def _create_sqlite_backend(config: Dict[str, Any]) -> SQLiteBackend:
    """Create SQLite backend from config."""
    db_path = config.get(
        "db_path",
        os.getenv("KESTREL_DB_PATH", "./agent_data/kestrel.db")
    )
    return SQLiteBackend(db_path)


def _create_postgres_backend(config: Dict[str, Any]) -> DatabaseBackend:
    """Create PostgreSQL backend from config.

    ADVANCED MODE: PostgreSQL is for multi-tenant, high-concurrency, or server
    deployments. For sovereign agents, use SQLiteBackend (the default).
    """
    from .postgres import PostgresBackend as _PostgresBackend

    # Try DSN first
    dsn = config.get("dsn") or os.getenv("KESTREL_DATABASE_URL")
    if dsn:
        return _PostgresBackend(
            dsn=dsn,
            min_pool_size=config.get("min_pool_size", 2),
            max_pool_size=config.get("max_pool_size", 10),
        )
    
    # Fall back to individual parameters
    return _PostgresBackend(
        host=config.get("host") or os.getenv("KESTREL_DB_HOST", "localhost"),
        port=int(config.get("port") or os.getenv("KESTREL_DB_PORT", "5432")),
        database=config.get("database") or os.getenv("KESTREL_DB_NAME", "kestrel"),
        user=config.get("user") or os.getenv("KESTREL_DB_USER"),
        password=config.get("password") or os.getenv("KESTREL_DB_PASSWORD"),
        min_pool_size=config.get("min_pool_size", 2),
        max_pool_size=config.get("max_pool_size", 10),
    )
