"""
SQLite-First Sync Layer

This module provides synchronization capabilities for SQLite databases,
enabling cloud backup and optional PostgreSQL aggregation while maintaining
SQLite as the primary runtime.

Architecture:
    SQLite (primary) ---> Sync Layer ---> Cloud/PostgreSQL (secondary)

Components:
    - WALListener: Monitors SQLite WAL for changes
    - SyncService: Orchestrates replication to targets
    - CloudTarget: Abstracts cloud storage destinations (S3, Lighthouse)
    - PostgresTarget: Optional PostgreSQL sync target

Usage:
    from kestrel_sovereign.storage.sync import SyncService, WALListener

    # Create sync service with SQLite database
    sync = SyncService(db_path="/path/to/agent.db")

    # Add sync targets
    sync.add_target(S3Target(bucket="my-bucket"))

    # Start continuous sync
    await sync.start()

Design Principles:
    1. SQLite is the source of truth
    2. Sync is asynchronous and non-blocking
    3. Targets are optional and pluggable
    4. Offline-first: works without network
"""

from kestrel_sovereign.storage.sync.wal_listener import WALListener
from kestrel_sovereign.storage.sync.service import SyncService
from kestrel_sovereign.storage.sync.targets import SyncTarget, S3Target, LighthouseTarget

__all__ = [
    "WALListener",
    "SyncService",
    "SyncTarget",
    "S3Target",
    "LighthouseTarget",
]
