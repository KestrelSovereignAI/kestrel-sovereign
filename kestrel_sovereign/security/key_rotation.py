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


def _master_bytes_from_key(key: str) -> bytes:
    """Resolve a raw key string to its 44-byte URL-safe-base64 master form.

    Mirrors ``get_master_key_bytes()`` exactly so per-agent HKDF derivations
    stay identical across runtime and rotation. If the input is a
    Fernet-shaped key (44-byte URL-safe base64), it's used directly;
    otherwise it's treated as a passphrase and SHA-256-derived. Any
    drift between this function and ``get_master_key_bytes`` corrupts
    every per-agent row during rotation.
    """
    from cryptography.fernet import Fernet  # shape probe only
    import base64
    try:
        Fernet(key)  # raises if not a real Fernet-shaped key
    except Exception:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest)
    return key.encode("ascii") if isinstance(key, str) else key


def _derive_agent_cipher(master: bytes, agent_id: str) -> "AEADCipher":
    """Derive a per-agent ``AEADCipher`` from master bytes and agent_id.

    Mirrors ``get_agent_fernet`` byte-for-byte (HKDF-SHA256 with
    ``salt=agent_id`` and ``info=b"kestrel-agent-v1"``). Used by rotation
    to derive matching ciphers for both old and new masters when walking
    per-agent encrypted rows in ``conversation_history``.
    """
    import base64
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=agent_id.encode("utf-8"),
        info=b"kestrel-agent-v1",
    )
    derived = hkdf.derive(master)
    return AEADCipher(base64.urlsafe_b64encode(derived))


def _validate_key(key: str) -> "AEADCipher":
    """Validate that a key can be used as an AEADCipher key.

    The key-derivation logic here MUST match ``get_fernet()`` exactly: any
    divergence means rotation encrypts new rows with a different key than
    the runtime decrypts with — rotated rows become unreadable after the
    user swaps ``KESTREL_DATA_KEY``.

    Logic mirrors ``get_fernet()``:

    1. If the input is a real Fernet-shaped key (44-byte URL-safe base64
       encoding 32 raw bytes), use it directly. ``Fernet(key)`` is the
       authoritative shape detector — ``AEADCipher(key)`` would accept
       any 32-byte input (e.g. a 32-character passphrase) as a raw AES
       key, which would diverge from ``get_fernet()``'s passphrase path.
    2. Otherwise treat the input as a passphrase and derive via SHA-256.
    """
    return AEADCipher(_master_bytes_from_key(key))


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

        # Validate the new key (raises if it can't be coerced to a usable cipher)
        _validate_key(new_key)
        new_master = _master_bytes_from_key(new_key)

        # Get current key
        from kestrel_sovereign.security.encryption import _get_data_key
        old_key = _get_data_key()
        if not old_key:
            raise ValueError("No current key configured (KESTREL_DATA_KEY)")
        old_master = _master_bytes_from_key(old_key)

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
            self._execute_rotation(rotation, old_master, new_master),
            rotation.id,
        )

        return rotation.id

    async def get_status(self, rotation_id: str) -> Optional[RotationRecord]:
        """Get the status of a rotation operation."""
        row = await self.storage.database.fetchone(
            "SELECT * FROM key_rotations WHERE id = ?",
            (rotation_id,),
        )
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
        from kestrel_sovereign.security.encryption import _get_data_key
        old_key = _get_data_key()
        if not old_key:
            raise ValueError("No current key configured (KESTREL_DATA_KEY)")
        old_master = _master_bytes_from_key(old_key)
        new_master = _master_bytes_from_key(new_key)

        self._current_rotation = rotation
        self._track_rotation_task(
            self._execute_rotation(rotation, old_master, new_master),
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

    # ------------------------------------------------------------------
    # Encrypted-table registry
    # ------------------------------------------------------------------
    #
    # Tables that store encrypted content the rotation must walk. Names
    # match the production storage layer (``conversation_history`` —
    # NOT the legacy ``conversations`` — and ``files``). Adding a new
    # encrypted table elsewhere in the codebase requires a one-line
    # entry here so the rotation actually walks it; the previous
    # hardcoded list silently drifted from production for years.

    # Each entry declares whether the table uses per-agent HKDF-derived
    # encryption (``agent_id_column`` is set) or global encryption
    # (``agent_id_column = None``). Per-agent rows are encrypted under
    # ``HKDF(master, salt=agent_id, info=b"kestrel-agent-v1")``; rotation
    # MUST derive the matching cipher per row, otherwise the global cipher
    # cannot decrypt those rows and they're silently skipped — corrupting
    # data after a key swap.
    ENCRYPTED_TABLES: List[Dict[str, Optional[str]]] = [
        {
            "table": "conversation_history",
            "content_column": "content",
            "id_column": "id",
            "agent_id_column": "agent_id",
        },
        {
            "table": "files",
            "content_column": "content",
            "id_column": "content_hash",
            "agent_id_column": None,
        },
    ]

    async def _execute_rotation(
        self,
        rotation: RotationRecord,
        old_master: bytes,
        new_master: bytes,
    ):
        """Execute the actual key rotation.

        Takes 44-byte URL-safe-base64 master keys for both sides so each
        per-agent row can have its HKDF-derived cipher built on demand.
        Same-key (master-equal) runs short-circuit ``LIKE 'KSAv2:%'``
        rows since they're already in v2 — useful for the
        ``upgrade_to_aead`` path.
        """
        same_key = old_master == new_master
        try:
            for entry in self.ENCRYPTED_TABLES:
                await self._rotate_table(
                    rotation, old_master, new_master,
                    table=entry["table"],
                    content_column=entry["content_column"],
                    id_column=entry["id_column"],
                    agent_id_column=entry.get("agent_id_column"),
                    skip_v2=same_key,
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
        old_master: bytes,
        new_master: bytes,
        table: str,
        content_column: str,
        id_column: str,
        agent_id_column: Optional[str] = None,
        skip_v2: bool = False,
    ):
        """Rotate encryption for a single table.

        Handles three orthogonal correctness concerns from the #936 review:

        1. **Per-agent HKDF rows.** When ``agent_id_column`` is set, each
           row's cipher pair is derived from its ``agent_id`` and the
           old/new masters. This is what ``conversation_history`` needs;
           a global-cipher rotation would silently fail to decrypt these
           rows and lose them on the post-rotation key swap.
        2. **BLOB vs TEXT content columns.** ``files.content`` is BLOB,
           ``conversation_history.content`` is TEXT. Read both as either
           ``bytes`` or ``str``, normalize for the cipher, and write back
           in the same shape we read.
        3. **UPDATE + progress INSERT atomicity.** Wrapped in a
           transaction so a crash between them cannot leave the row
           rewritten under ``new_master`` with ``rotation_progress``
           silently missing it.

        ``skip_v2=True`` (set by ``_execute_rotation`` when masters are
        equal) narrows the SELECT to ``gAAAAA%`` only — same-key upgrade
        runs do not need to revisit already-v2 rows.
        """
        already_rotated_rows = await self.storage.database.fetchall(
            "SELECT record_id FROM rotation_progress WHERE rotation_id = ? AND table_name = ?",
            (rotation.id, table),
        )
        already_rotated = {row[0] for row in already_rotated_rows}

        safe_tbl = safe_table_name(table)
        safe_id_col = safe_column_name(id_column)
        safe_content_col = safe_column_name(content_column)
        safe_agent_col = safe_column_name(agent_id_column) if agent_id_column else None

        if skip_v2:
            where = f"{safe_content_col} LIKE 'gAAAAA%'"
        else:
            where = (
                f"{safe_content_col} LIKE 'gAAAAA%' "
                f"   OR {safe_content_col} LIKE 'KSAv2:%'"
            )

        if safe_agent_col:
            select_cols = f"{safe_id_col}, {safe_content_col}, {safe_agent_col}"
        else:
            select_cols = f"{safe_id_col}, {safe_content_col}"

        rows = await self.storage.database.fetchall(
            f"SELECT {select_cols} FROM {safe_tbl} WHERE {where}"
        )

        # Pre-build the global cipher pair; per-agent ciphers are derived per row
        old_global = AEADCipher(old_master)
        new_global = AEADCipher(new_master)

        # Memoize per-agent cipher pairs to avoid re-running HKDF for every row
        agent_cipher_cache: Dict[str, tuple] = {}

        consecutive_failures = 0
        for row in rows:
            record_id = row[0]
            encrypted_content = row[1]
            agent_id = row[2] if safe_agent_col else None

            record_id_str = str(record_id)
            if record_id_str in already_rotated:
                continue

            # Pick the right NEW cipher (encrypt side). For per-agent tables
            # we always write under the per-agent HKDF — matches production's
            # ``add_conversation`` (key_version >= 1).
            if agent_id and safe_agent_col:
                pair = agent_cipher_cache.get(agent_id)
                if pair is None:
                    pair = (
                        _derive_agent_cipher(old_master, agent_id),
                        _derive_agent_cipher(new_master, agent_id),
                    )
                    agent_cipher_cache[agent_id] = pair
                old_agent_cipher, new_cipher = pair
                # Decrypt-side fallback list: per-agent first, then global —
                # matches ``_decrypt_with_fallback`` in async_conversation_store.
                # This handles mixed corpora where some rows were written with
                # the global cipher (key_version 0 / pre-v1).
                decrypt_candidates = [old_agent_cipher, old_global]
            else:
                new_cipher = new_global
                decrypt_candidates = [old_global]

            # Normalize bytes/str — file rows are BLOB, conversation rows TEXT
            was_str = isinstance(encrypted_content, str)
            ct_bytes = encrypted_content.encode() if was_str else encrypted_content

            try:
                decrypted = None
                last_err: Optional[Exception] = None
                for cand in decrypt_candidates:
                    try:
                        decrypted = cand.decrypt(ct_bytes)
                        break
                    except Exception as inner:
                        last_err = inner
                if decrypted is None:
                    raise last_err if last_err else RuntimeError("no decrypt candidate")

                new_ct_bytes = new_cipher.encrypt(decrypted)
                new_value = new_ct_bytes.decode() if was_str else new_ct_bytes

                # UPDATE + progress INSERT atomic
                async with self.storage.database.transaction():
                    await self.storage.database.execute(
                        f"UPDATE {safe_tbl} SET {safe_content_col} = ? "
                        f"WHERE {safe_id_col} = ?",
                        (new_value, record_id),
                    )
                    await self.storage.database.execute(
                        "INSERT INTO rotation_progress "
                        "(rotation_id, table_name, record_id, rotated_at) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            rotation.id, table, record_id_str,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )

                rotation.records_processed += 1
                consecutive_failures = 0

                if rotation.records_processed % 100 == 0:
                    await self._save_rotation(rotation)
                    logger.info(
                        f"Rotation progress: "
                        f"{rotation.records_processed}/{rotation.records_total}"
                    )

            except Exception as e:
                consecutive_failures += 1
                logger.warning(
                    f"Failed to rotate record {record_id} in {table} "
                    f"(agent_id={agent_id}): {e}"
                )
                # Abort the run if the first 5 rows all fail — that smells
                # like a key-derivation bug, not isolated row corruption.
                # Keeps a regression like the #931 passphrase-derivation
                # mismatch from silently completing with records_processed=0.
                if consecutive_failures >= 5 and rotation.records_processed == 0:
                    raise RuntimeError(
                        f"Rotation aborting: first {consecutive_failures} rows "
                        f"in {table} all failed to decrypt. Likely a key-"
                        f"derivation drift between rotation and runtime. Last "
                        f"error: {e}"
                    ) from e

    async def _count_encrypted_records(self) -> int:
        """Count total encrypted records across all registered tables."""
        total = 0
        for entry in self.ENCRYPTED_TABLES:
            table = entry["table"]
            column = entry["content_column"]
            try:
                safe_tbl = safe_table_name(table)
                safe_col = safe_column_name(column)
                value = await self.storage.database.fetchval(
                    f"SELECT COUNT(*) FROM {safe_tbl} "
                    f"WHERE {safe_col} LIKE 'gAAAAA%' "
                    f"   OR {safe_col} LIKE 'KSAv2:%'"
                )
                total += value or 0
            except Exception as e:
                # Table may not exist in this storage backend
                logger.warning(f"Could not count encrypted data in {table}.{column}: {e}")
                continue
        return total

    # ------------------------------------------------------------------
    # AEAD upgrade — same key, lift v1 Fernet rows to v2 AES-256-GCM
    # ------------------------------------------------------------------

    async def upgrade_to_aead(
        self,
        new_key: Optional[str] = None,
        new_key_file: Optional[str] = None,
    ) -> str:
        """Re-encrypt every Fernet row as v2 AEAD without changing the key.

        Wave 0C (#915) makes new writes emit ``KSAv2:`` v2 tokens; the
        AEADCipher legacy-decrypt path keeps existing Fernet rows readable.
        That is sufficient for ongoing operation, but a full corpus stays
        on the weaker AES-128 primitive until each row is touched. This
        method drives an eager re-encryption sweep using the current
        ``KESTREL_DATA_KEY`` for both sides — the same key, just a
        stronger format.

        Optionally the caller can pass ``new_key`` / ``new_key_file`` to
        rotate AND upgrade in one pass; otherwise the runtime key is used.

        Returns the rotation id.
        """
        if new_key is None and new_key_file is None:
            from kestrel_sovereign.security.encryption import _get_data_key
            current = _get_data_key()
            if not current:
                raise ValueError("KESTREL_DATA_KEY is not configured")
            new_key = current
        return await self.start_rotation(new_key=new_key, new_key_file=new_key_file)

    async def _get_in_progress_rotation(self) -> Optional[RotationRecord]:
        """Check for any in-progress rotation."""
        row = await self.storage.database.fetchone(
            "SELECT * FROM key_rotations WHERE status = ?",
            (RotationStatus.IN_PROGRESS.value,),
        )
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
