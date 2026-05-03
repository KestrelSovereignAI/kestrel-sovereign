"""
Key Rotation Service

Allows changing the master encryption key (KESTREL_DATA_KEY) without
losing access to encrypted data.

Usage:
    from kestrel_sovereign.security.key_rotation import KeyRotationService

    service = KeyRotationService(storage)
    rotation_id = await service.start_rotation("/path/to/new_key")
    status = await service.get_status(rotation_id)
"""

import asyncio
import hashlib
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional

from kestrel_sdk.security.aead import AEADCipher

from kestrel_sovereign.sql_utils import safe_table_name, safe_column_name
from kestrel_sovereign.security.encryption import DecryptionError, get_fernet, _read_key_from_file

logger = logging.getLogger(__name__)


class RotationStatus(Enum):
    """Status of a key rotation operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class RotationRecord:
    """Tracks a key rotation operation."""
    id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    old_key_hash: str = ""
    new_key_hash: str = ""
    status: RotationStatus = RotationStatus.PENDING
    records_processed: int = 0
    records_total: int = 0
    error_message: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.status in (RotationStatus.COMPLETED, RotationStatus.FAILED, RotationStatus.ROLLED_BACK)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "old_key_hash": self.old_key_hash,
            "new_key_hash": self.new_key_hash,
            "status": self.status.value,
            "records_processed": self.records_processed,
            "records_total": self.records_total,
            "error_message": self.error_message,
        }


def _hash_key(key: str) -> str:
    """Create a hash of a key for storage (don't store the key itself)."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _validate_key(key: str) -> "AEADCipher":
    """Validate that a key can be used as an AEADCipher key.

    Returns an ``AEADCipher`` (drop-in for the legacy ``Fernet`` return type;
    AES-256-GCM with Fernet read-compat per Wave 0C of the Quantum Hardening
    epic).
    """
    from kestrel_sdk.security.aead import AEADCipher
    try:
        # Try as raw Fernet key (44-byte URL-safe base64)
        return AEADCipher(key)
    except Exception:
        # Derive from passphrase
        digest = hashlib.sha256(key.encode('utf-8')).digest()
        import base64
        return AEADCipher(base64.urlsafe_b64encode(digest))


class KeyRotationService:
    """
    Service for rotating encryption keys.

    This allows changing KESTREL_DATA_KEY without losing access to
    encrypted data. All encrypted records are re-encrypted with
    the new key.
    """

    def __init__(self, storage):
        """
        Initialize the key rotation service.

        Args:
            storage: AsyncStorage instance with database access
        """
        self.storage = storage
        self._current_rotation: Optional[RotationRecord] = None
        self._rotation_tasks: set[asyncio.Task[None]] = set()

    async def initialize(self):
        """Create rotation tracking tables if they don't exist."""
        await self.storage.database.execute("""
            CREATE TABLE IF NOT EXISTS key_rotations (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                old_key_hash TEXT NOT NULL,
                new_key_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                records_processed INTEGER DEFAULT 0,
                records_total INTEGER DEFAULT 0,
                error_message TEXT
            )
        """)

        await self.storage.database.execute("""
            CREATE TABLE IF NOT EXISTS rotation_progress (
                rotation_id TEXT NOT NULL,
                table_name TEXT NOT NULL,
                record_id TEXT NOT NULL,
                rotated_at TEXT NOT NULL,
                PRIMARY KEY (rotation_id, table_name, record_id)
            )
        """)

    async def start_rotation(
        self,
        new_key_file: Optional[str] = None,
        new_key: Optional[str] = None,
    ) -> str:
        """
        Start a key rotation operation.

        Args:
            new_key_file: Path to file containing the new key
            new_key: The new key directly (prefer file for security)

        Returns:
            Rotation ID for tracking progress

        Raises:
            ValueError: If no new key provided or key is invalid
            RuntimeError: If a rotation is already in progress
        """
        # Check for in-progress rotation
        existing = await self._get_in_progress_rotation()
        if existing:
            raise RuntimeError(
                f"Rotation {existing.id} is already in progress. "
                "Resume or rollback before starting new rotation."
            )

        # Get new key
        if new_key_file:
            new_key = _read_key_from_file(new_key_file)
        if not new_key:
            raise ValueError("No new key provided")

        # Validate new key
        new_fernet = _validate_key(new_key)

        # Get current key
        old_fernet = get_fernet()
        if not old_fernet:
            raise ValueError("No current key configured (KESTREL_DATA_KEY)")

        # Get current key for hashing
        from kestrel_sovereign.security.encryption import _get_data_key
        old_key = _get_data_key()
        if not old_key:
            raise ValueError("Cannot retrieve current key for hashing")

        # Create rotation record
        rotation = RotationRecord(
            id=str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc),
            old_key_hash=_hash_key(old_key),
            new_key_hash=_hash_key(new_key),
            status=RotationStatus.IN_PROGRESS,
        )

        # Count records to rotate
        rotation.records_total = await self._count_encrypted_records()

        # Store rotation record
        await self._save_rotation(rotation)
        self._current_rotation = rotation

        logger.info(
            f"Starting key rotation {rotation.id}: "
            f"{rotation.records_total} records to process"
        )

        self._track_rotation_task(
            self._execute_rotation(rotation, old_fernet, new_fernet),
            rotation.id,
        )

        return rotation.id

    async def get_status(self, rotation_id: str) -> Optional[RotationRecord]:
        """Get the status of a rotation operation."""
        async with self.storage.database.execute(
            "SELECT * FROM key_rotations WHERE id = ?",
            (rotation_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return RotationRecord(
                    id=row[0],
                    started_at=datetime.fromisoformat(row[1]),
                    completed_at=datetime.fromisoformat(row[2]) if row[2] else None,
                    old_key_hash=row[3],
                    new_key_hash=row[4],
                    status=RotationStatus(row[5]),
                    records_processed=row[6],
                    records_total=row[7],
                    error_message=row[8],
                )
        return None

    async def resume_rotation(self, rotation_id: str, key_file: str) -> bool:
        """
        Resume an interrupted rotation.

        Args:
            rotation_id: ID of the rotation to resume
            key_file: Path to the key file (new key being rotated to)

        Returns:
            True if resumed successfully
        """
        rotation = await self.get_status(rotation_id)
        if not rotation:
            raise ValueError(f"Rotation {rotation_id} not found")

        if rotation.status != RotationStatus.IN_PROGRESS:
            raise ValueError(f"Rotation {rotation_id} is not in progress")

        new_key = _read_key_from_file(key_file)
        if not new_key or _hash_key(new_key) != rotation.new_key_hash:
            raise ValueError("Key does not match rotation's new key")

        # Continue from where we left off
        old_fernet = get_fernet()
        new_fernet = _validate_key(new_key)

        self._current_rotation = rotation
        self._track_rotation_task(
            self._execute_rotation(rotation, old_fernet, new_fernet),
            rotation.id,
        )

        return True

    def _track_rotation_task(
        self,
        coro: Coroutine[Any, Any, None],
        rotation_id: str,
    ) -> asyncio.Task[None]:
        """Own background rotation tasks so shutdown cannot orphan DB writes."""
        task = asyncio.create_task(coro, name=f"key-rotation-{rotation_id}")
        self._rotation_tasks.add(task)
        task.add_done_callback(self._rotation_tasks.discard)
        return task

    async def drain_rotations(self, *, cancel: bool = False) -> None:
        """Wait for tracked rotation tasks, optionally cancelling for shutdown."""
        tasks = set(self._rotation_tasks)
        if not tasks:
            return

        if cancel:
            for task in tasks:
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        self._rotation_tasks.difference_update(tasks)

    async def shutdown(self) -> None:
        """
        Stop owned rotation tasks before storage closes.

        Cancelled rotations remain in-progress in the tracking table and can be
        resumed through the existing resume path.
        """
        await self.drain_rotations(cancel=True)

    async def _execute_rotation(
        self,
        rotation: RotationRecord,
        old_fernet: "AEADCipher",
        new_fernet: "AEADCipher"
    ):
        """Execute the actual key rotation."""
        try:
            # Rotate conversation messages
            await self._rotate_table(
                rotation, old_fernet, new_fernet,
                table="conversations",
                content_column="content",
                id_column="rowid"
            )

            # Rotate file store
            await self._rotate_table(
                rotation, old_fernet, new_fernet,
                table="files",
                content_column="content",
                id_column="file_id"
            )

            # Rotation complete
            rotation.status = RotationStatus.COMPLETED
            rotation.completed_at = datetime.now(timezone.utc)
            await self._save_rotation(rotation)

            logger.info(
                f"Key rotation {rotation.id} completed: "
                f"{rotation.records_processed} records rotated"
            )

        except Exception as e:
            rotation.status = RotationStatus.FAILED
            rotation.error_message = str(e)
            rotation.completed_at = datetime.now(timezone.utc)
            await self._save_rotation(rotation)
            logger.error(f"Key rotation {rotation.id} failed: {e}")

    async def _rotate_table(
        self,
        rotation: RotationRecord,
        old_fernet: "AEADCipher",
        new_fernet: "AEADCipher",
        table: str,
        content_column: str,
        id_column: str,
    ):
        """Rotate encryption for a single table."""
        # Get records not yet rotated
        already_rotated = set()
        async with self.storage.database.execute(
            "SELECT record_id FROM rotation_progress WHERE rotation_id = ? AND table_name = ?",
            (rotation.id, table)
        ) as cursor:
            async for row in cursor:
                already_rotated.add(row[0])

        # Validate identifiers for safe SQL interpolation
        safe_tbl = safe_table_name(table)
        safe_id_col = safe_column_name(id_column)
        safe_content_col = safe_column_name(content_column)

        # Get all records
        async with self.storage.database.execute(
            f"SELECT {safe_id_col}, {safe_content_col} FROM {safe_tbl} WHERE {safe_content_col} LIKE 'gAAAAA%'"
        ) as cursor:
            rows = await cursor.fetchall()

        for record_id, encrypted_content in rows:
            record_id_str = str(record_id)
            if record_id_str in already_rotated:
                continue

            try:
                # Decrypt with old key
                decrypted = old_fernet.decrypt(encrypted_content.encode())

                # Re-encrypt with new key
                new_encrypted = new_fernet.encrypt(decrypted).decode()

                # Update record
                await self.storage.database.execute(
                    f"UPDATE {safe_tbl} SET {safe_content_col} = ? WHERE {safe_id_col} = ?",
                    (new_encrypted, record_id)
                )

                # Mark as rotated
                await self.storage.database.execute(
                    "INSERT INTO rotation_progress (rotation_id, table_name, record_id, rotated_at) VALUES (?, ?, ?, ?)",
                    (rotation.id, table, record_id_str, datetime.now(timezone.utc).isoformat())
                )

                rotation.records_processed += 1

                # Update progress periodically
                if rotation.records_processed % 100 == 0:
                    await self._save_rotation(rotation)
                    logger.info(f"Rotation progress: {rotation.records_processed}/{rotation.records_total}")

            except Exception as e:
                logger.warning(f"Failed to rotate record {record_id} in {table}: {e}")
                # Continue with other records

    async def _count_encrypted_records(self) -> int:
        """Count total encrypted records across all tables."""
        total = 0

        for table, column in [("conversations", "content"), ("files", "content")]:
            try:
                safe_tbl = safe_table_name(table)
                safe_col = safe_column_name(column)
                async with self.storage.database.execute(
                    f"SELECT COUNT(*) FROM {safe_tbl} WHERE {safe_col} LIKE 'gAAAAA%'"
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        total += row[0]
            except Exception as e:
                # Table may not exist in this storage backend
                logger.warning(f"Could not count encrypted data in {table}.{column}: {e}")
                continue

        return total

    async def _get_in_progress_rotation(self) -> Optional[RotationRecord]:
        """Check for any in-progress rotation."""
        async with self.storage.database.execute(
            "SELECT * FROM key_rotations WHERE status = ?",
            (RotationStatus.IN_PROGRESS.value,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return RotationRecord(
                    id=row[0],
                    started_at=datetime.fromisoformat(row[1]),
                    old_key_hash=row[3],
                    new_key_hash=row[4],
                    status=RotationStatus(row[5]),
                    records_processed=row[6],
                    records_total=row[7],
                )
        return None

    async def _save_rotation(self, rotation: RotationRecord):
        """Save rotation record to database."""
        await self.storage.database.execute("""
            INSERT OR REPLACE INTO key_rotations
            (id, started_at, completed_at, old_key_hash, new_key_hash, status, records_processed, records_total, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rotation.id,
            rotation.started_at.isoformat(),
            rotation.completed_at.isoformat() if rotation.completed_at else None,
            rotation.old_key_hash,
            rotation.new_key_hash,
            rotation.status.value,
            rotation.records_processed,
            rotation.records_total,
            rotation.error_message,
        ))


# CLI entry point
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Kestrel Key Rotation")
    subparsers = parser.add_subparsers(dest="command")

    # rotate command
    rotate_parser = subparsers.add_parser("rotate", help="Start key rotation")
    rotate_parser.add_argument("--new-key-file", required=True, help="Path to new key file")
    rotate_parser.add_argument("--db-path", default=".", help="Path to agent database")

    # status command
    status_parser = subparsers.add_parser("status", help="Check rotation status")
    status_parser.add_argument("--rotation-id", help="Specific rotation to check")
    status_parser.add_argument("--db-path", default=".", help="Path to agent database")

    # resume command
    resume_parser = subparsers.add_parser("resume", help="Resume interrupted rotation")
    resume_parser.add_argument("--rotation-id", required=True, help="Rotation ID to resume")
    resume_parser.add_argument("--key-file", required=True, help="Path to key file")
    resume_parser.add_argument("--db-path", default=".", help="Path to agent database")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    print(f"Command: {args.command}")
    print("Note: Full implementation requires async runtime. Use from Python code.")
