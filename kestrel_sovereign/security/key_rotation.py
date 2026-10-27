"""
Key Rotation Service

Re-encrypts registered stores for a master-key (KESTREL_DATA_KEY) change.
The host owns maintenance isolation and key cutover; unregistered stores
are outside this service's coverage.

The host owns one rotation service for a database and must quiesce legacy
master-key-encrypted writers during rotation. Keep both original keys until
COMPLETED; resume a failed or interrupted journal before changing the runtime
key or starting another partially overlapping rotation. This module does not
provide a distributed maintenance lease or an online writer cutover.

Usage:
    from kestrel_sovereign.security.key_rotation import KeyRotationService

    service = KeyRotationService(storage)
    rotation_id = await service.start_rotation("/path/to/new_key")
    status = await service.get_status(rotation_id)
"""

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Coroutine, Dict, List, Optional

from kestrel_sdk.security.aead import AEADCipher

from kestrel_sovereign.sql_utils import safe_table_name, safe_column_name
from kestrel_sovereign.security.encryption import (
    # get_fernet is not called in this module's own code: it is a patch seam.
    # tests/unit/test_key_rotation.py does
    # ``monkeypatch.setattr(key_rotation, "get_fernet", ...)``, which needs the
    # name bound here. The redundant alias marks it as intentional so an
    # unused-import sweep cannot strip it.
    get_fernet as get_fernet,
    _read_key_from_file,
)

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

    Re-encrypt registered encrypted records under a new master key.
    The host owns maintenance isolation and key cutover; unregistered stores
    and the known salted-passphrase limitation are outside this service.
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
        self._rotation_admission_lock = asyncio.Lock()

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
        async with self._rotation_admission_lock:
            await self._require_rotation_admission()

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

    async def _require_rotation_admission(self, resume_id: str | None = None) -> None:
        """One admission rule for every start/resume, under the service lock.

        A retiring task owns the database even after its status was saved.
        A partially applied journal owns the key transition after task exit.
        Only that same journal may resume; another historical failed attempt
        cannot claim its remaining old-key rows.
        """
        if any(not task.done() for task in self._rotation_tasks):
            raise RuntimeError("A rotation is already running; wait for its owner to finish")
        existing = await self._get_in_progress_rotation(excluding_id=resume_id)
        if existing:
            raise RuntimeError(
                f"Rotation {existing.id} is unresolved. "
                "Resume with its original keys before starting a new rotation."
            )

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
        Resume an interrupted or failed rotation with its original key pair.

        Args:
            rotation_id: ID of the rotation to resume
            key_file: Path to the key file (new key being rotated to)

        Returns:
            True if resumed successfully
        """
        async with self._rotation_admission_lock:
            await self._require_rotation_admission(rotation_id)
            rotation = await self.get_status(rotation_id)
            if not rotation:
                raise ValueError(f"Rotation {rotation_id} not found")

            if rotation.status not in {RotationStatus.IN_PROGRESS, RotationStatus.FAILED}:
                raise ValueError(f"Rotation {rotation_id} is not resumable")

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

            if _hash_key(old_key) != rotation.old_key_hash:
                raise ValueError("Current key does not match rotation's old key")
            # Progress rows commit atomically with their ciphertext. Counters may
            # lag after cancellation, so recover them from that durable journal.
            row = await self.storage.database.fetchone(
                "SELECT COUNT(*) FROM rotation_progress WHERE rotation_id = ?", (rotation_id,))
            rotation.records_processed = row[0]
            rotation.status = RotationStatus.IN_PROGRESS
            rotation.completed_at = None
            rotation.error_message = None
            await self._save_rotation(rotation)
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

    # Each entry declares:
    # - ``agent_id_column``: per-agent HKDF table (set) vs global (None).
    #   Per-agent rows are encrypted under
    #   ``HKDF(master, salt=agent_id, info=b"kestrel-agent-v1")``; rotation
    #   MUST derive the matching cipher per row, or the global cipher fails
    #   to decrypt and rows are silently skipped — corruption on key swap.
    # - ``optional_content``: a nullable companion column whose absent value
    #   has no ciphertext even when the row-level metadata declares encryption.
    # Selection follows the runtime's metadata.enc flag, never token prefixes.
    ENCRYPTED_TABLES: List[Dict[str, Any]] = [
        {
            "table": "conversation_history",
            "content_column": "content",
            "id_column": "id",
            "agent_id_column": "agent_id",
        },
        # rendered_content (#1402) holds the byte-stable transport form
        # encrypted under the same per-agent fernet as ``content``. It
        # gets a separate registry entry so rotation walks it in its own
        # pass; ``progress_alias`` keeps the rotation_progress key
        # distinct from the ``content`` pass for the same row.
        {
            "table": "conversation_history",
            "content_column": "rendered_content",
            "id_column": "id",
            "agent_id_column": "agent_id",
            "progress_alias": "conversation_history.rendered_content",
            "optional_content": True,
        },
        {
            "table": "files",
            "content_column": "content",
            "id_column": "content_hash",
            "agent_id_column": None,
        },
        {
            # Saved artifacts use envelope encryption; rotating the wrapped
            # index/content root leaves their owner/blob addresses stable.
            "table": "artifact_keys",
            "content_column": "wrapped_key",
            "id_column": "key_id",
            "agent_id_column": None,
            "optional": True,
            "require_all": True,
        },
        {
            # Visual leases/connections/actions retain the same stable envelope
            # through offline master-key maintenance as saved artifacts.
            "table": "visual_computer_keys",
            "content_column": "wrapped_key",
            "id_column": "key_id",
            "agent_id_column": None,
            "optional": True,
            "require_all": True,
        },
    ]

    async def _entry_available(self, entry: Dict[str, Any]) -> bool:
        if not entry.get("optional"):
            return True
        db = self.storage.database
        if db.backend_type == "postgres":
            row = await db.fetchone("SELECT to_regclass(?) IS NOT NULL", (entry["table"],))
            return bool(row and row[0])
        row = await db.fetchone("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (entry["table"],))
        return row is not None

    async def _execute_rotation(
        self,
        rotation: RotationRecord,
        old_master: bytes,
        new_master: bytes,
    ):
        """Execute the actual key rotation.

        Takes 44-byte URL-safe-base64 master keys for both sides so each
        per-agent row can have its HKDF-derived cipher built on demand.
        Same-key upgrade runs verify and rewrite all metadata-flagged rows,
        including existing AEAD ciphertext, rather than trusting a prefix.
        """
        try:
            for entry in self.ENCRYPTED_TABLES:
                if not await self._entry_available(entry):
                    continue
                await self._rotate_table(
                    rotation, old_master, new_master,
                    table=entry["table"],
                    content_column=entry["content_column"],
                    id_column=entry["id_column"],
                    agent_id_column=entry.get("agent_id_column"),
                    optional_content=entry.get("optional_content", False),
                    progress_alias=entry.get("progress_alias"),
                    require_all=entry.get("require_all", False),
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
        optional_content: bool = False,
        progress_alias: Optional[str] = None,
        require_all: bool = False,
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

        ``progress_alias`` (#1402) lets two passes over the same SQL
        table (one per encrypted column) record progress under distinct
        rotation_progress keys. SQL queries still use ``table``; the
        progress table sees ``progress_alias or table``.

        Every selected encrypted row must succeed, including a verified write
        and atomic progress receipt. A failure retains a resumable journal.
        ``require_all`` additionally makes envelope-root entries mandatory:
        malformed/empty key tables and invalid resumed roots cannot be skipped.
        """
        progress_key = progress_alias or table
        already_rotated_rows = await self.storage.database.fetchall(
            "SELECT record_id FROM rotation_progress WHERE rotation_id = ? AND table_name = ?",
            (rotation.id, progress_key),
        )
        already_rotated = {row[0] for row in already_rotated_rows}

        safe_tbl = safe_table_name(table)
        safe_id_col = safe_column_name(id_column)
        safe_content_col = safe_column_name(content_column)
        safe_agent_col = safe_column_name(agent_id_column) if agent_id_column else None

        rows = await self._encrypted_rows(
            table=table, content_column=content_column, id_column=id_column,
            agent_id_column=agent_id_column, require_all=require_all,
            optional_content=optional_content,
        )

        # Pre-build the global cipher pair; per-agent ciphers are derived per row
        old_global = AEADCipher(old_master)
        new_global = AEADCipher(new_master)

        # Memoize per-agent cipher pairs to avoid re-running HKDF for every row
        agent_cipher_cache: Dict[str, tuple] = {}

        for row in rows:
            record_id = row[0]
            encrypted_content = row[1]
            agent_id = row[2] if safe_agent_col else None

            record_id_str = str(record_id)
            if record_id_str in already_rotated:
                if require_all:
                    value = encrypted_content.encode() if isinstance(encrypted_content, str) else encrypted_content
                    new_global.decrypt(value)
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
                    affected = await self.storage.database.execute(
                        f"UPDATE {safe_tbl} SET {safe_content_col} = ? "
                        f"WHERE {safe_id_col} = ?",
                        (new_value, record_id),
                    )
                    stored = await self.storage.database.fetchone(
                        f"SELECT {safe_content_col} FROM {safe_tbl} WHERE {safe_id_col} = ?", (record_id,))
                    if affected != 1 or stored is None or stored[0] != new_value:
                        raise RuntimeError(f"Required encrypted record in {table} was not updated")
                    await self.storage.database.execute(
                        "INSERT INTO rotation_progress "
                        "(rotation_id, table_name, record_id, rotated_at) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            rotation.id, progress_key, record_id_str,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )

                rotation.records_processed += 1

                if rotation.records_processed % 100 == 0:
                    await self._save_rotation(rotation)
                    logger.info(
                        f"Rotation progress: "
                        f"{rotation.records_processed}/{rotation.records_total}"
                    )

            except Exception as e:
                # A skipped encrypted row would remain under the old key even
                # if the overall run reported success. Preserve partial progress
                # and require an explicit resume with both original keys instead.
                raise RuntimeError(f"Encrypted record {record_id_str[:128]!r} in {table} could not be rotated") from e

    async def _encrypted_rows(
        self, *, table: str, content_column: str, id_column: str,
        agent_id_column: str | None = None, require_all: bool = False,
        optional_content: bool = False,
    ) -> list:
        """Use the runtime's encryption marker for both counting and rotation.

        A ciphertext-looking prefix is valid plaintext when metadata.enc is
        absent/false. Conversely, a damaged flagged ciphertext remains required
        even when its prefix is missing. Envelope tables have no plaintext rows.
        """
        safe_tbl = safe_table_name(table)
        safe_col = safe_column_name(content_column)
        safe_id = safe_column_name(id_column)
        agent = safe_column_name(agent_id_column) if agent_id_column else "NULL"
        columns = f"{safe_id}, {safe_col}, {agent}"
        if not require_all:
            columns += ", metadata"
        rows = await self.storage.database.fetchall(
            f"SELECT {columns} FROM {safe_tbl} ORDER BY {safe_id}"
        )
        selected = []
        for row in rows:
            if not require_all:
                try:
                    metadata = json.loads(row[3]) if row[3] else None
                    if not metadata or not metadata.get("enc"):
                        continue
                except (ValueError, TypeError, AttributeError) as exc:
                    raise RuntimeError(
                        f"Invalid encryption metadata for record {str(row[0])[:128]!r} in {table}"
                    ) from exc
                if optional_content and row[1] is None:
                    continue
            selected.append(row[:3])
        if require_all and not selected:
            raise RuntimeError(f"Required encrypted key table {table} is empty")
        return selected

    async def _count_encrypted_records(self) -> int:
        """Count exactly the same metadata-selected records as the writer."""
        total = 0
        for entry in self.ENCRYPTED_TABLES:
            if not await self._entry_available(entry):
                continue
            total += len(await self._encrypted_rows(
                table=entry["table"], content_column=entry["content_column"],
                id_column=entry["id_column"], agent_id_column=entry.get("agent_id_column"),
                require_all=entry.get("require_all", False),
                optional_content=entry.get("optional_content", False),
            ))
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

    async def _get_in_progress_rotation(self, *, excluding_id: str | None = None) -> Optional[RotationRecord]:
        """Find any unresolved journal other than the exact resume target."""
        sql = ("SELECT * FROM key_rotations k WHERE (k.status = ? OR (k.status = ? "
               "AND EXISTS (SELECT 1 FROM rotation_progress p WHERE p.rotation_id = k.id)))")
        params = [RotationStatus.IN_PROGRESS.value, RotationStatus.FAILED.value]
        if excluding_id is not None:
            sql += " AND k.id != ?"
            params.append(excluding_id)
        row = await self.storage.database.fetchone(sql + " ORDER BY started_at LIMIT 1", tuple(params))
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
