"""
Async Conversation Store for Kestrel Storage.

Provides async conversation history management with encryption at rest.
Encryption is enabled via KESTREL_DATA_KEY environment variable.

Key Versioning:
    - key_version: 0 (or missing) = global key (backward compat)
    - key_version: 1 = per-agent HKDF-derived key

All queries are scoped by agent_id for multi-tenant isolation.
"""
import json
import logging
import os
from typing import Dict, Optional, List, Any

from .async_database import AsyncDatabase
from .conversation_ids import coerce_persistent_message_id
from .encryption import (
    get_fernet, get_agent_fernet, encrypt_string, decrypt_string, remove_enc_flag,
    DecryptionError
)

logger = logging.getLogger(__name__)

# Current key version for new encryptions
CURRENT_KEY_VERSION = 1


def _rows_affected(result) -> int:
    """Normalise the return value of AsyncDatabase.execute_commit (#763).

    Different backends return slightly different shapes:
      * SQLite/Postgres backends return ``cursor.rowcount`` (int).
      * Some legacy paths returned a Result-like object with ``.rowcount``.

    Soft-delete needs an honest count so callers can distinguish "row
    was already in trash" from "row was just trashed." The pre-existing
    ``hasattr(result, 'rowcount') else True`` fallback lied (always
    True), which broke the no-op semantics. Use this helper instead.
    """
    if isinstance(result, int):
        return result
    if hasattr(result, "rowcount") and result.rowcount is not None:
        return result.rowcount
    return 0


class AsyncConversationStore:
    """Async conversation history storage with per-agent encryption."""

    def __init__(self, db: AsyncDatabase, agent_id: str = ""):
        self.db = db
        self.agent_id = agent_id
        # Global key for backward compatibility
        self._global_fernet = get_fernet()
        # Per-agent key (recommended, used for new data)
        self._agent_fernet = get_agent_fernet(agent_id) if agent_id else None
        # Auto-migration on read (can be disabled via env var)
        self._migrate_on_read = os.environ.get("KESTREL_DISABLE_MIGRATION") != "true"

    def _now_sql(self) -> str:
        """Get SQL expression for current timestamp based on backend type."""
        if self.db.backend_type == "postgres":
            return "NOW()"
        return "datetime('now')"

    @property
    def encryption_enabled(self) -> bool:
        """Check if encryption at rest is enabled."""
        return self._agent_fernet is not None or self._global_fernet is not None

    @property
    def _fernet(self):
        """Backward compatibility - return agent fernet or global."""
        return self._agent_fernet or self._global_fernet

    # Session boundary constant: see kestrel_sdk.config.constants
    @property
    def _IMPLICIT_SESSION_GAP_MINUTES(self) -> int:
        from kestrel_sdk.config.constants import SESSION_GAP_MINUTES
        return SESSION_GAP_MINUTES

    async def _derive_implicit_session_id(self) -> Optional[str]:
        """
        Derive an implicit session_id from the time-gap heuristic.

        Returns the previous message's session_id if it was within
        the last 30 minutes; otherwise mints a new UUID for a new
        implicit session.

        This makes the implicit session boundaries already used by
        MemoryConsolidator and wellness metrics observable in metadata,
        so callers that filter by session_id get sensible groupings
        even when no explicit session_id is provided.
        """
        try:
            row = await self.db.fetchone(
                "SELECT metadata, created_at FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL "
                "ORDER BY id DESC LIMIT 1",
                (self.agent_id,),
            )
            if not row:
                # First message ever — start a new session
                return self._new_session_id()

            prev_metadata_str, prev_created_at = row
            prev_meta = json.loads(prev_metadata_str) if prev_metadata_str else {}

            # If the previous message has no session_id (legacy data), start fresh
            prev_sid = prev_meta.get("session_id")
            if not prev_sid:
                return self._new_session_id()

            # Compare gap; reuse if within window
            from datetime import datetime, timezone, timedelta
            if isinstance(prev_created_at, str):
                try:
                    prev_dt = datetime.fromisoformat(prev_created_at.replace("Z", "+00:00"))
                except ValueError:
                    return self._new_session_id()
            else:
                return self._new_session_id()

            if prev_dt.tzinfo is None:
                prev_dt = prev_dt.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            gap = now - prev_dt
            if gap < timedelta(minutes=self._IMPLICIT_SESSION_GAP_MINUTES):
                return prev_sid
            return self._new_session_id()
        except Exception as e:
            # Never let implicit-session derivation block the write
            logger.warning(f"Implicit session derivation failed: {e}")
            return None

    @staticmethod
    def _new_session_id() -> str:
        """Mint a new implicit session_id (UUID4)."""
        import uuid
        return str(uuid.uuid4())

    async def add_conversation(self, role: str, content: str,
                               metadata: Optional[Dict] = None,
                               session_id: Optional[str] = None) -> None:
        """Add a conversation message with per-agent encryption.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata dict
            session_id: If provided, link this message to a specific session.
                       This allows resuming old conversations beyond the 30-min gap.
                       If not provided, an implicit session_id is derived from
                       the time-gap heuristic (30 min inactivity = new session).
        """
        meta = dict(metadata) if metadata else {}

        # Resolve session_id: explicit wins; otherwise derive from time gap
        if not session_id:
            session_id = await self._derive_implicit_session_id()

        if session_id:
            meta['session_id'] = session_id

        # Use per-agent key for new messages
        fernet_to_use = self._agent_fernet or self._global_fernet
        to_store, was_encrypted = encrypt_string(content, fernet_to_use)

        if was_encrypted:
            meta['enc'] = True
            meta['key_version'] = CURRENT_KEY_VERSION

        await self.db.execute_commit(
            f"INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, {self._now_sql()})",
            (self.agent_id, role, to_store, json.dumps(meta) if meta else None)
        )

    def _decrypt_with_fallback(self, content: str, meta: Optional[Dict]) -> tuple[str, bool]:
        """Decrypt content, trying per-agent key first then global.

        Returns:
            Tuple of (decrypted_content, needs_migration)

        Raises:
            DecryptionError: If content is marked as encrypted but all
                           decryption attempts fail (wrong key)
        """
        if not meta or not meta.get("enc"):
            return content, False

        key_version = meta.get("key_version", 0)
        last_error: Optional[DecryptionError] = None

        # Version 1+: use per-agent key directly
        if key_version >= 1 and self._agent_fernet:
            try:
                return decrypt_string(content, meta, self._agent_fernet), False
            except DecryptionError as e:
                last_error = e  # Fall through to global key

        # Version 0 or fallback: try global key
        if self._global_fernet:
            try:
                decrypted = decrypt_string(content, meta, self._global_fernet)
                # If we decrypted with global key but have agent key, needs migration
                needs_migration = (
                    key_version == 0 and
                    self._agent_fernet is not None and
                    decrypted != content
                )
                return decrypted, needs_migration
            except DecryptionError as e:
                last_error = e

        # Last resort: try agent key even for old version (maybe re-encrypted)
        if self._agent_fernet:
            try:
                return decrypt_string(content, meta, self._agent_fernet), False
            except DecryptionError as e:
                last_error = e

        # All decryption attempts failed - raise error
        error_msg = f"Failed to decrypt message for agent {self.agent_id}"
        logger.error(error_msg)
        if last_error:
            raise DecryptionError(error_msg) from last_error
        else:
            raise DecryptionError(f"{error_msg}: No encryption keys available")

    async def _migrate_message(self, row_id: int, decrypted_content: str) -> None:
        """Re-encrypt a message with per-agent key."""
        if not self._agent_fernet:
            return

        new_content, _ = encrypt_string(decrypted_content, self._agent_fernet)
        new_meta = {"enc": True, "key_version": CURRENT_KEY_VERSION}

        await self.db.execute_commit(
            "UPDATE conversation_history SET content = ?, metadata = ? WHERE id = ?",
            (new_content, json.dumps(new_meta), row_id)
        )
        logger.debug(f"Migrated message {row_id} to per-agent encryption")

    async def get_conversation_history(
        self, limit: int = 100, session_id: str = None
    ) -> List[Dict[str, Any]]:
        """Get recent conversation history with automatic decryption and migration.

        Args:
            limit: Maximum number of messages to return
            session_id: If provided, get messages from this session only (using time-based grouping)
        """
        if session_id:
            # Session-aware retrieval: get messages from the specified session
            rows = await self._get_session_messages(session_id, limit)
        else:
            # Default behavior: get most recent live messages
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, created_at FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL "
                "ORDER BY id DESC LIMIT ?",
                (self.agent_id, limit)
            )
        history = []
        for row in reversed(rows):  # Return in chronological order
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else None

            # Skip messages excluded from context (compressed, summarized, etc.)
            if meta and meta.get("excluded_from_context"):
                continue

            content, needs_migration = self._decrypt_with_fallback(row[2], meta)

            # Opportunistic migration to per-agent key
            if needs_migration and self._migrate_on_read:
                try:
                    await self._migrate_message(row_id, content)
                except Exception as e:
                    logger.warning(f"Migration failed for message {row_id}: {e}")

            entry = {
                'id': row_id,
                'role': row[1],
                'content': content,
                'created_at': row[4]
            }
            cleaned_meta = remove_enc_flag(meta)
            if cleaned_meta:
                # Remove internal key_version from external metadata
                cleaned_meta.pop('key_version', None)
                if cleaned_meta:
                    entry['metadata'] = cleaned_meta
            history.append(entry)
        return history

    @staticmethod
    def _deleted_filter_clause(deleted_filter: str) -> str:
        """Return the SQL fragment that filters by deleted_at state.

        ``live``    → only ``deleted_at IS NULL`` (default for reads).
        ``deleted`` → only ``deleted_at IS NOT NULL`` (Trash view, restore).
        ``all``     → no filter (purge needs every row, regardless of state).

        Returns the leading ``AND`` so it can be appended to a WHERE
        clause that already has at least one condition. ``all`` returns
        an empty string.
        """
        if deleted_filter == "live":
            return " AND deleted_at IS NULL"
        if deleted_filter == "deleted":
            return " AND deleted_at IS NOT NULL"
        if deleted_filter == "all":
            return ""
        raise ValueError(
            f"Invalid deleted_filter={deleted_filter!r}; "
            "expected 'live', 'deleted', or 'all'"
        )

    async def _get_session_messages(
        self,
        session_id: str,
        limit: int,
        deleted_filter: str = "live",
    ) -> List[tuple]:
        """Get messages belonging to a specific session.

        Sessions are determined by:
        1. Time-based grouping (30-minute gaps end a session)
        2. Explicit session_id in metadata (for resumed conversations)

        The session_id is the message ID that starts that session.

        Args:
            session_id: The message ID that marks the session start
            limit: Maximum messages to return
            deleted_filter: ``live`` (default — for reads),
                ``deleted`` (for restore / Trash view), or
                ``all`` (for purge — finds rows in any state).

        Returns:
            List of raw rows (id, role, content, metadata, created_at)
        """
        del_clause = self._deleted_filter_clause(deleted_filter)
        from datetime import datetime
        from kestrel_sdk.config.constants import SESSION_GAP_MINUTES

        # Try to interpret session_id as a message ID for time-based grouping.
        # If it isn't (e.g. a UUID-based implicit session_id), skip this path
        # and fall through to the metadata-based lookup below.
        all_rows = []
        row_id = coerce_persistent_message_id(session_id)
        if row_id is not None:
            # The anchor row itself is looked up regardless of state — we
            # need its timestamp even if it's been soft-deleted, otherwise
            # we can't restore the session that owned it.
            start_row = await self.db.fetchone(
                "SELECT created_at FROM conversation_history WHERE id = ? AND agent_id = ?",
                (row_id, self.agent_id)
            )

            # If session_id is a message ID, get messages from that timestamp forward
            if start_row:
                start_time = start_row[0]
                all_rows = await self.db.fetchall(
                    f"""SELECT id, role, content, metadata, created_at
                       FROM conversation_history
                       WHERE agent_id = ? AND created_at >= ?{del_clause}
                       ORDER BY created_at ASC
                       LIMIT ?""",
                    (self.agent_id, start_time, limit * 2)  # Fetch extra in case of filtering
                )

        # Also get messages that explicitly belong to this session (resumed conversations)
        # These are messages with session_id in metadata that may come after a time gap
        resumed_rows = await self.db.fetchall(
            f"""SELECT id, role, content, metadata, created_at
               FROM conversation_history
               WHERE agent_id = ? AND metadata LIKE ?{del_clause}
               ORDER BY created_at ASC
               LIMIT ?""",
            (self.agent_id, f'%"session_id": "{session_id}"%', limit)
        )

        # Also try without space after colon (JSON formatting varies)
        resumed_rows_alt = await self.db.fetchall(
            f"""SELECT id, role, content, metadata, created_at
               FROM conversation_history
               WHERE agent_id = ? AND metadata LIKE ?{del_clause}
               ORDER BY created_at ASC
               LIMIT ?""",
            (self.agent_id, f'%"session_id":"{session_id}"%', limit)
        )

        # Merge resumed rows (dedupe by id)
        seen_ids = set()
        merged_rows = []
        for row in all_rows + resumed_rows + resumed_rows_alt:
            if row[0] not in seen_ids:
                seen_ids.add(row[0])
                merged_rows.append(row)

        # Sort by created_at
        merged_rows.sort(key=lambda r: r[4] or '')

        # Filter to only include messages in this session
        session_rows = []
        last_timestamp = None
        is_first = True
        session_id_str = str(session_id)

        for row in merged_rows:
            created_at = row[4]
            metadata_json = row[3]

            # Parse timestamp
            if isinstance(created_at, str):
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                    try:
                        timestamp = datetime.strptime(created_at, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    timestamp = datetime.now()
            elif created_at:
                timestamp = created_at
            else:
                timestamp = datetime.now()

            # Check if this message explicitly belongs to this session (resumed)
            is_resumed_message = False
            if metadata_json:
                try:
                    meta = json.loads(metadata_json)
                    if meta.get('session_id') == session_id_str:
                        is_resumed_message = True
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata for message in session {session_id}: {e}")

            # Check for new_session marker (skip after first message, but not for resumed)
            if not is_first and not is_resumed_message and metadata_json:
                try:
                    meta = json.loads(metadata_json)
                    if meta.get('new_session'):
                        break
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata for new_session check: {e}")

            # Check time gap (only for non-resumed messages)
            if last_timestamp and not is_resumed_message:
                gap_minutes = (timestamp - last_timestamp).total_seconds() / 60
                if gap_minutes > SESSION_GAP_MINUTES:
                    # Skip this non-resumed message (it belongs to a different session)
                    # Continue looking for resumed messages that explicitly belong to this session
                    continue

            # Skip session markers from results
            if metadata_json:
                try:
                    meta = json.loads(metadata_json)
                    if meta.get('type') == 'session_marker':
                        last_timestamp = timestamp
                        is_first = False
                        continue
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata for session_marker check: {e}")

            session_rows.append(row)
            last_timestamp = timestamp
            is_first = False

            if len(session_rows) >= limit:
                break

        # Return in DESC order to match the non-session query format
        # (will be reversed by caller in get_conversation_history)
        return list(reversed(session_rows))

    async def get_full_history(self) -> List[Dict[str, Any]]:
        """Get complete live conversation history with automatic decryption.

        Soft-deleted rows (#763) are filtered out — use
        ``get_full_history_with_ids(include_deleted=True)`` if you need
        to see Trash too.
        """
        rows = await self.db.fetchall(
            "SELECT id, role, content, metadata FROM conversation_history "
            "WHERE agent_id = ? AND deleted_at IS NULL "
            "ORDER BY id ASC",
            (self.agent_id,)
        )
        history = []
        for row in rows:
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else None
            content, needs_migration = self._decrypt_with_fallback(row[2], meta)

            # Opportunistic migration
            if needs_migration and self._migrate_on_read:
                try:
                    await self._migrate_message(row_id, content)
                except Exception as e:
                    logger.warning(f"Migration failed for message {row_id} in get_full_history: {e}")

            cleaned_meta = remove_enc_flag(meta)
            if cleaned_meta:
                cleaned_meta.pop('key_version', None)

            entry = {
                'role': row[1],
                'content': content,
                'metadata': cleaned_meta if cleaned_meta else None
            }
            history.append(entry)
        return history

    async def search_history(
        self,
        query: str,
        limit: int = 20,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search conversation history.

        Fetches and decrypts messages, then filters client-side.
        This approach works correctly with encrypted storage.

        Args:
            query: Substring to search for (case-insensitive).
            limit: Maximum results to return.
            session_id: If provided, restrict search to messages tagged
                with this session_id in metadata. Useful for "what did
                we discuss in this session" queries.
        """
        # SQL pre-filter when session_id is provided. We can match against
        # the metadata JSON because session_id is plaintext (not encrypted).
        # Falls back to full scan when no session_id is given.
        if session_id:
            # Match both `"session_id": "X"` and `"session_id":"X"` formats
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL "
                "AND (metadata LIKE ? OR metadata LIKE ?) "
                "ORDER BY id DESC LIMIT 5000",
                (
                    self.agent_id,
                    f'%"session_id": "{session_id}"%',
                    f'%"session_id":"{session_id}"%',
                ),
            )
        else:
            # Fetch all live messages (up to 5000) and search client-side after decryption
            # SQL LIKE doesn't work on encrypted content, so we must decrypt first
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL "
                "ORDER BY id DESC LIMIT 5000",
                (self.agent_id,)
            )

        query_lower = query.lower()
        results = []

        for row in rows:
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else None
            content, needs_migration = self._decrypt_with_fallback(row[2], meta)

            # Opportunistic migration
            if needs_migration and self._migrate_on_read:
                try:
                    await self._migrate_message(row_id, content)
                except Exception as e:
                    logger.warning(f"Migration failed for message {row_id} in search_history: {e}")

            # Client-side search on decrypted content
            if query_lower in content.lower():
                cleaned_meta = remove_enc_flag(meta)
                if cleaned_meta:
                    cleaned_meta.pop('key_version', None)

                results.append({
                    'role': row[1],
                    'content': content,
                    'metadata': cleaned_meta if cleaned_meta else None
                })

                if len(results) >= limit:
                    break

        return results

    async def clear_history(self) -> None:
        """Soft-delete every live message for this agent (#763).

        Stamps ``deleted_at`` instead of issuing a SQL DELETE so the rows
        remain recoverable from Trash until the retention janitor (#764)
        sweeps them. Already-deleted rows are left alone — re-stamping
        would extend their retention window.

        Use ``purge_all`` when you genuinely need to destroy the rows
        (administrative wipe, EPHEMERAL session close, restore-from-CAR).
        """
        await self.db.execute_commit(
            f"UPDATE conversation_history SET deleted_at = {self._now_sql()} "
            "WHERE agent_id = ? AND deleted_at IS NULL",
            (self.agent_id,)
        )

    # ------------------------------------------------------------------
    # Conversation titles (user-assigned rename support — issue #716).
    # ------------------------------------------------------------------
    #
    # Stored out-of-band in ``conversation_titles`` rather than on
    # conversation_history rows because:
    #   * message rows are frequently encrypted and their metadata JSON
    #     carries unrelated bookkeeping
    #   * rename is a single-row upsert here, vs. having to find-and-edit
    #     the first message of a session
    #   * deleting a session is a separate concern (#715) that wipes
    #     messages; the title row is stale-but-harmless until the user
    #     does something explicit to address it

    MAX_CONVERSATION_NAME_LENGTH = 120

    async def set_conversation_name(
        self, session_id: str, name: Optional[str]
    ) -> Optional[str]:
        """Upsert the user-chosen display name for a conversation.

        Args:
            session_id: The session whose name we're setting.
            name: New name.  ``None`` or empty-after-strip clears the
                  override (the UI will fall back to the computed
                  preview).  Non-empty values are trimmed and capped to
                  ``MAX_CONVERSATION_NAME_LENGTH``.

        Returns:
            The final stored value (trimmed), or ``None`` when cleared.
        """
        # Normalize inputs.  Empty strings and whitespace-only strings
        # collapse to "clear the override" so UI callers can just blur
        # an empty text input and get the expected behavior.
        if name is None:
            stored: Optional[str] = None
        else:
            trimmed = name.strip()
            if not trimmed:
                stored = None
            else:
                stored = trimmed[: self.MAX_CONVERSATION_NAME_LENGTH]

        if stored is None:
            await self.db.execute_commit(
                "DELETE FROM conversation_titles "
                "WHERE agent_id = ? AND session_id = ?",
                (self.agent_id, session_id),
            )
            return None

        # Upsert.  SQLite and Postgres both accept the ON CONFLICT syntax.
        await self.db.execute_commit(
            "INSERT INTO conversation_titles (agent_id, session_id, name, updated_at) "
            f"VALUES (?, ?, ?, {self._now_sql()}) "
            "ON CONFLICT (agent_id, session_id) DO UPDATE SET "
            f"  name = excluded.name, updated_at = {self._now_sql()}",
            (self.agent_id, session_id, stored),
        )
        return stored

    async def get_conversation_name(self, session_id: str) -> Optional[str]:
        """Return the user-assigned name for a conversation, or None."""
        row = await self.db.fetchone(
            "SELECT name FROM conversation_titles "
            "WHERE agent_id = ? AND session_id = ?",
            (self.agent_id, session_id),
        )
        if not row:
            return None
        return row[0]

    async def get_conversation_names(self) -> Dict[str, str]:
        """Return {session_id: name} for every titled session owned by
        this agent.  Used by the list endpoint to decorate the response
        in a single round-trip instead of querying per row.
        """
        rows = await self.db.fetchall(
            "SELECT session_id, name FROM conversation_titles "
            "WHERE agent_id = ? AND name IS NOT NULL",
            (self.agent_id,),
        )
        return {row[0]: row[1] for row in rows if row[1]}

    async def delete_message(self, message_id: int) -> bool:
        """Soft-delete a specific message by ID (#763).

        Stamps ``deleted_at`` so the row survives in Trash until purged
        explicitly or aged out by the retention janitor (#764). A row
        that's already soft-deleted reports ``False`` (no-op) rather than
        re-stamping its deleted_at.

        Returns:
            True if a live row was soft-deleted, False if not found or
            already in trash.
        """
        affected = await self.db.execute_commit(
            f"UPDATE conversation_history SET deleted_at = {self._now_sql()} "
            "WHERE id = ? AND agent_id = ? AND deleted_at IS NULL",
            (message_id, self.agent_id)
        )
        return _rows_affected(affected) > 0

    async def delete_conversation_session(self, session_id: str) -> int:
        """Soft-delete every live message in the given session (#763).

        Resolves which messages belong to `session_id` using the same
        logic `_get_session_messages` uses for loading — which covers
        both explicit UUID-based sessions (session_id stored in message
        metadata JSON) and legacy time-gap-based sessions (session_id
        is the row id of the first message in the cluster, and cluster
        members are discovered by time-gap walking).

        Stamps ``deleted_at`` rather than issuing a hard DELETE so the
        session is recoverable from Trash. Use ``purge_session`` for
        permanent removal.

        Args:
            session_id: The session to soft-delete.  Accepts either a
                UUID string (for metadata-based sessions) or a numeric
                message ID (for legacy time-gap sessions).

        Returns:
            Number of live messages stamped.  Returns 0 if the session
            doesn't exist, isn't owned by this agent, or is empty / all
            already soft-deleted.

        Notes:
            Per-agent scoped via the `agent_id = ?` filter in the final
            UPDATE.  Already-soft-deleted rows are not re-stamped (their
            existing deleted_at controls retention).  Ephemeral-mode
            callers must be rejected at the privacy wrapper above —
            this method does not read the privacy config.
        """
        rows = await self._get_session_messages(session_id, limit=10_000)
        if not rows:
            return 0

        ids = [row[0] for row in rows]
        if not ids:
            return 0

        placeholders = ",".join("?" for _ in ids)
        params = [*ids, self.agent_id]
        affected = await self.db.execute_commit(
            f"UPDATE conversation_history "
            f"SET deleted_at = {self._now_sql()} "
            f"WHERE id IN ({placeholders}) AND agent_id = ? "
            f"AND deleted_at IS NULL",
            tuple(params),
        )
        return _rows_affected(affected)

    # ------------------------------------------------------------------
    # Restore primitives (#763 / #765)
    # ------------------------------------------------------------------
    #
    # Mirror image of the soft-delete methods. Clear ``deleted_at`` so
    # the row reappears in normal reads. A row that was never soft-
    # deleted is a no-op (rowcount=0).

    async def restore_message(self, message_id: int) -> bool:
        """Clear deleted_at on a soft-deleted message (#763).

        Returns:
            True if a soft-deleted row was restored, False if the row
            doesn't exist, isn't owned by this agent, or wasn't actually
            in trash.
        """
        affected = await self.db.execute_commit(
            "UPDATE conversation_history SET deleted_at = NULL "
            "WHERE id = ? AND agent_id = ? AND deleted_at IS NOT NULL",
            (message_id, self.agent_id),
        )
        return _rows_affected(affected) > 0

    async def restore_conversation_session(self, session_id: str) -> int:
        """Clear deleted_at on every soft-deleted message in a session.

        Uses the same session-resolution logic as soft-delete but with
        ``deleted_filter='deleted'`` so we find messages that are in
        trash, not the live ones.

        Returns:
            Number of rows restored. Zero if the session has no soft-
            deleted rows or doesn't exist.
        """
        rows = await self._get_session_messages(
            session_id, limit=10_000, deleted_filter="deleted"
        )
        if not rows:
            return 0

        ids = [row[0] for row in rows]
        if not ids:
            return 0

        placeholders = ",".join("?" for _ in ids)
        params = [*ids, self.agent_id]
        affected = await self.db.execute_commit(
            f"UPDATE conversation_history SET deleted_at = NULL "
            f"WHERE id IN ({placeholders}) AND agent_id = ? "
            f"AND deleted_at IS NOT NULL",
            tuple(params),
        )
        return _rows_affected(affected)

    # ------------------------------------------------------------------
    # Purge primitives (#763)
    # ------------------------------------------------------------------
    #
    # Hard SQL DELETE — the row is gone, no recovery. Callers must
    # supply a reason string for the audit trail (#750). Privacy mode
    # enforcement happens at the wrapper layer above.

    async def purge_message(
        self, message_id: int, reason: str = "user-initiated"
    ) -> bool:
        """Hard-delete a single message (#763).

        Removes the row regardless of whether it's currently live or
        already soft-deleted. The ``reason`` argument is recorded by
        the caller in the audit log; this method just performs the
        DELETE.

        Returns:
            True if a row was destroyed, False if not found.
        """
        affected = await self.db.execute_commit(
            "DELETE FROM conversation_history WHERE id = ? AND agent_id = ?",
            (message_id, self.agent_id),
        )
        deleted = _rows_affected(affected) > 0
        if deleted:
            logger.info(
                "purge_message id=%s agent=%s reason=%s",
                message_id, self.agent_id, reason,
            )
        return deleted

    async def purge_conversation_session(
        self, session_id: str, reason: str = "user-initiated"
    ) -> int:
        """Hard-delete every message in a session, live or soft-deleted (#763).

        Uses ``deleted_filter='all'`` so we find both live messages and
        ones that previously soft-deleted into trash. The whole session
        is destroyed in one transaction.

        Returns:
            Number of rows destroyed.
        """
        rows = await self._get_session_messages(
            session_id, limit=10_000, deleted_filter="all"
        )
        if not rows:
            return 0

        ids = [row[0] for row in rows]
        if not ids:
            return 0

        placeholders = ",".join("?" for _ in ids)
        params = [*ids, self.agent_id]
        affected = await self.db.execute_commit(
            f"DELETE FROM conversation_history "
            f"WHERE id IN ({placeholders}) AND agent_id = ?",
            tuple(params),
        )
        purged = _rows_affected(affected)
        if purged:
            logger.info(
                "purge_conversation_session sid=%s agent=%s reason=%s rows=%d",
                session_id, self.agent_id, reason, purged,
            )
        return purged

    async def purge_all(self, reason: str = "administrative") -> int:
        """Hard-delete every conversation row for this agent (#763).

        Reserved for restore-from-CAR and explicit administrative wipe.
        NOT the user-facing 'clear history' button — that goes through
        ``clear_history``.  NOT the EPHEMERAL leak-purge — that path
        calls :meth:`purge_all_since` with the timestamp the agent
        entered EPHEMERAL so it can only destroy rows written *during*
        the EPHEMERAL stint.  Calling this method on a long-lived agent
        wipes the entire history regardless of when rows were authored
        (#867).
        """
        affected = await self.db.execute_commit(
            "DELETE FROM conversation_history WHERE agent_id = ?",
            (self.agent_id,),
        )
        purged = _rows_affected(affected)
        logger.info(
            "purge_all agent=%s reason=%s rows=%d",
            self.agent_id, reason, purged,
        )
        return purged

    async def purge_all_since(
        self,
        since_iso: str,
        *,
        reason: str = "ephemeral-leak",
    ) -> int:
        """Hard-delete conversation rows authored on/after ``since_iso``.

        Scoped variant of :meth:`purge_all` for the EPHEMERAL leak-purge
        (#867).  EPHEMERAL is "leave no trace," so anything written
        *during* the stint is a privacy-layer leak — but rows authored
        before the agent entered EPHEMERAL are preexisting NORMAL data
        the user explicitly wanted persisted.  Only rows whose
        ``created_at >= since_iso`` are destroyed.  ``since_iso`` is
        captured at the moment the wrapper sees the transition INTO
        EPHEMERAL.

        If ``since_iso`` is empty/None, returns 0 without running the
        DELETE — the absence of a timestamp means we can't safely scope,
        and the original wipe-on-shutdown bug is precisely what this
        method exists to prevent.
        """
        if not since_iso:
            logger.warning(
                "purge_all_since called without since_iso — refusing to purge "
                "(agent=%s, reason=%s)",
                self.agent_id, reason,
            )
            return 0
        affected = await self.db.execute_commit(
            "DELETE FROM conversation_history "
            "WHERE agent_id = ? AND created_at >= ?",
            (self.agent_id, since_iso),
        )
        purged = _rows_affected(affected)
        logger.info(
            "purge_all_since agent=%s since=%s reason=%s rows=%d",
            self.agent_id, since_iso, reason, purged,
        )
        return purged

    async def purge_trash_older_than(
        self,
        cutoff_iso: str,
        *,
        max_rows: int = 10_000,
        reason: str = "retention-janitor",
    ) -> int:
        """Hard-delete soft-deleted rows older than ``cutoff_iso`` (#764).

        The retention janitor calls this on a periodic tick to enforce
        the per-agent retention window. Three safety rails layered into
        one query:

        1. ``deleted_at IS NOT NULL`` — live rows are NEVER touched, no
           matter how old. The janitor's job is to age out trash, not
           data the user is still using.
        2. ``deleted_at < ?`` — the cutoff. Caller computes
           ``now - retention_days`` once per sweep so all rows in a
           batch use the same threshold.
        3. ``LIMIT ?`` (via ``IN (subquery)``) — prevents a runaway
           sweep from stalling the writer thread for minutes if the
           agent suddenly has 500k aged rows. The janitor calls back
           on the next tick to drain the rest.

        Args:
            cutoff_iso: ISO-8601 timestamp string. Rows whose
                ``deleted_at`` is strictly less than this are eligible
                for purge.
            max_rows: Hard cap on rows destroyed in a single call.
                Defaults to 10k. Set lower for tests.
            reason: Audit reason; lands in the operator log.

        Returns:
            Number of rows actually destroyed.
        """
        if not cutoff_iso:
            return 0
        if max_rows <= 0:
            return 0

        # SQLite doesn't support LIMIT directly inside DELETE on every
        # build path, and even when it does the syntax differs from
        # PostgreSQL. The IN (SELECT ... LIMIT ...) form is portable.
        affected = await self.db.execute_commit(
            "DELETE FROM conversation_history "
            "WHERE id IN ("
            "  SELECT id FROM conversation_history "
            "  WHERE agent_id = ? "
            "    AND deleted_at IS NOT NULL "
            "    AND deleted_at < ? "
            "  ORDER BY deleted_at ASC "
            "  LIMIT ?"
            ")",
            (self.agent_id, cutoff_iso, max_rows),
        )
        purged = _rows_affected(affected)
        if purged:
            logger.info(
                "purge_trash_older_than agent=%s cutoff=%s reason=%s rows=%d",
                self.agent_id, cutoff_iso, reason, purged,
            )
        return purged

    async def delete_messages_matching(self, content_pattern: str) -> int:
        """Delete messages containing a specific pattern (case-insensitive).

        WARNING: This searches decrypted content, so it loads all messages first.
        Use carefully on large histories.

        Args:
            content_pattern: Text pattern to match in message content

        Returns:
            Number of messages deleted
        """
        # Get all messages with IDs
        history = await self.get_full_history_with_ids(include_excluded=True, include_stashed=True)

        # Find matching IDs
        pattern_lower = content_pattern.lower()
        ids_to_delete = []
        for msg in history:
            if pattern_lower in msg.get("content", "").lower():
                ids_to_delete.append(msg["id"])

        # Delete them
        for msg_id in ids_to_delete:
            await self.delete_message(msg_id)

        return len(ids_to_delete)

    async def get_full_history_with_ids(
        self,
        include_excluded: bool = False,
        include_stashed: bool = False,
        include_deleted: bool = False,
        only_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get complete conversation history with message IDs.

        Args:
            include_excluded: If True, include messages marked as excluded from context.
            include_stashed: If True, include messages that are stashed.
            include_deleted: If True, include soft-deleted rows alongside
                live rows. Default False — Trash stays hidden.
            only_deleted: If True, return ONLY soft-deleted rows. Used by
                the Trash UI (#765). Implies ``include_deleted``.

        Returns:
            List of message dicts with 'id', 'role', 'content', 'metadata',
            'created_at', and 'deleted_at' (None for live rows).
        """
        if only_deleted:
            del_clause = " AND deleted_at IS NOT NULL"
        elif include_deleted:
            del_clause = ""
        else:
            del_clause = " AND deleted_at IS NULL"

        rows = await self.db.fetchall(
            f"SELECT id, role, content, metadata, created_at, deleted_at "
            f"FROM conversation_history "
            f"WHERE agent_id = ?{del_clause} ORDER BY id ASC",
            (self.agent_id,)
        )
        history = []
        for row in rows:
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else {}
            content, needs_migration = self._decrypt_with_fallback(row[2], meta)

            # Filter out excluded messages unless requested
            if not include_excluded and meta.get("excluded_from_context"):
                continue

            # Filter out stashed messages unless requested
            if not include_stashed and meta.get("stashed"):
                continue

            # Opportunistic migration
            if needs_migration and self._migrate_on_read:
                try:
                    await self._migrate_message(row_id, content)
                except Exception as e:
                    logger.warning(f"Migration failed for message {row_id}: {e}")

            cleaned_meta = remove_enc_flag(meta)
            if cleaned_meta:
                cleaned_meta.pop('key_version', None)

            entry = {
                'id': row_id,
                'role': row[1],
                'content': content,
                'metadata': cleaned_meta if cleaned_meta else {},
                'created_at': row[4],
                'deleted_at': row[5],
            }
            history.append(entry)
        return history

    async def update_message_metadata(
        self,
        message_id: int,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """Update metadata for a specific message using atomic JSON merge.

        Uses SQL-level json_patch (PostgreSQL) or a SELECT-then-UPDATE with
        optimistic locking (SQLite) to avoid race conditions when multiple
        callers update metadata on the same message concurrently.

        Args:
            message_id: The message ID to update
            metadata_updates: Dict of metadata fields to update (merged with existing)

        Returns:
            True if message was found and updated, False otherwise
        """
        updates_json = json.dumps(metadata_updates)

        if self.db.backend_type == "postgres":
            # PostgreSQL: atomic JSON merge via || operator
            # COALESCE handles NULL metadata columns gracefully
            result = await self.db.execute_commit(
                "UPDATE conversation_history "
                "SET metadata = COALESCE(metadata::jsonb, '{}'::jsonb) || ?::jsonb "
                "WHERE id = ? AND agent_id = ?",
                (updates_json, message_id, self.agent_id)
            )
            updated = result.rowcount > 0 if hasattr(result, 'rowcount') else True
            if not updated:
                logger.warning(f"Message {message_id} not found for agent {self.agent_id}")
            return updated
        else:
            # SQLite: SELECT-then-UPDATE (single-writer, no true race condition)
            row = await self.db.fetchone(
                "SELECT metadata FROM conversation_history WHERE id = ? AND agent_id = ?",
                (message_id, self.agent_id)
            )
            if not row:
                logger.warning(f"Message {message_id} not found for agent {self.agent_id}")
                return False

            current_meta = json.loads(row[0]) if row[0] else {}
            current_meta.update(metadata_updates)

            await self.db.execute_commit(
                "UPDATE conversation_history SET metadata = ? WHERE id = ? AND agent_id = ?",
                (json.dumps(current_meta), message_id, self.agent_id)
            )
            return True

    async def update_messages_metadata(
        self,
        message_ids: List[int],
        metadata_updates: Dict[str, Any]
    ) -> int:
        """Update metadata for multiple messages.

        Args:
            message_ids: List of message IDs to update
            metadata_updates: Dict of metadata fields to update (merged with existing)

        Returns:
            Number of messages successfully updated
        """
        updated_count = 0
        for msg_id in message_ids:
            if await self.update_message_metadata(msg_id, metadata_updates):
                updated_count += 1
        return updated_count

    async def get_messages_by_ids(
        self,
        message_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Get specific messages by their IDs.

        Args:
            message_ids: List of message IDs to retrieve

        Returns:
            List of message dicts with 'id', 'role', 'content', 'metadata', 'created_at'
        """
        if not message_ids:
            return []

        placeholders = ",".join("?" * len(message_ids))
        rows = await self.db.fetchall(
            f"SELECT id, role, content, metadata, created_at FROM conversation_history "
            f"WHERE id IN ({placeholders}) AND agent_id = ? "
            f"AND deleted_at IS NULL ORDER BY id ASC",
            (*message_ids, self.agent_id)
        )

        history = []
        for row in rows:
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else {}
            content, _ = self._decrypt_with_fallback(row[2], meta)

            cleaned_meta = remove_enc_flag(meta)
            if cleaned_meta:
                cleaned_meta.pop('key_version', None)

            entry = {
                'id': row_id,
                'role': row[1],
                'content': content,
                'metadata': cleaned_meta if cleaned_meta else {},
                'created_at': row[4]
            }
            history.append(entry)
        return history

    async def search_messages_by_content(
        self,
        query: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search messages and return with IDs (for context management).

        Note: For encrypted content, this searches the encrypted text.
        For reliable search with encryption, use get_full_history_with_ids
        and filter client-side.

        Args:
            query: Search query string
            limit: Maximum results

        Returns:
            List of matching messages with IDs
        """
        # Get all messages and search client-side (handles encryption)
        all_messages = await self.get_full_history_with_ids(include_excluded=True)
        query_lower = query.lower()

        matches = []
        for msg in all_messages:
            if query_lower in msg.get("content", "").lower():
                matches.append(msg)
                if len(matches) >= limit:
                    break

        return matches

    async def get_excluded_messages(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get messages that have been excluded from context.

        Args:
            limit: Maximum messages to return

        Returns:
            List of excluded messages with their metadata
        """
        rows = await self.db.fetchall(
            "SELECT id, role, content, metadata, created_at FROM conversation_history "
            "WHERE agent_id = ? AND deleted_at IS NULL "
            "AND metadata LIKE '%\"excluded_from_context\": true%' "
            "ORDER BY id DESC LIMIT ?",
            (self.agent_id, limit)
        )

        results = []
        for row in rows:
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else {}
            content, _ = self._decrypt_with_fallback(row[2], meta)

            cleaned_meta = remove_enc_flag(meta)
            if cleaned_meta:
                cleaned_meta.pop('key_version', None)

            results.append({
                'id': row_id,
                'role': row[1],
                'content': content,
                'metadata': cleaned_meta if cleaned_meta else {},
                'created_at': row[4]
            })
        return results

    async def get_stashed_messages(
        self,
        stash_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get messages that have been stashed.

        Args:
            stash_id: Optional specific stash ID to retrieve
            limit: Maximum messages to return

        Returns:
            List of stashed messages with their metadata
        """
        if stash_id:
            # Get specific stash
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, created_at FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL AND metadata LIKE ? "
                "ORDER BY id ASC LIMIT ?",
                (self.agent_id, f'%"stash_id": "{stash_id}"%', limit)
            )
        else:
            # Get all stashed messages
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, created_at FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL "
                "AND metadata LIKE '%\"stashed\": true%' "
                "ORDER BY id DESC LIMIT ?",
                (self.agent_id, limit)
            )

        results = []
        for row in rows:
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else {}
            content, _ = self._decrypt_with_fallback(row[2], meta)

            cleaned_meta = remove_enc_flag(meta)
            if cleaned_meta:
                cleaned_meta.pop('key_version', None)

            results.append({
                'id': row_id,
                'role': row[1],
                'content': content,
                'metadata': cleaned_meta if cleaned_meta else {},
                'created_at': row[4]
            })
        return results

    async def list_stashes(self) -> List[Dict[str, Any]]:
        """Get a list of all stashes with summary info.

        Returns:
            List of stash summaries with id, name, message_count, stashed_at
        """
        # Get all stashed messages
        rows = await self.db.fetchall(
            "SELECT metadata FROM conversation_history "
            "WHERE agent_id = ? AND deleted_at IS NULL "
            "AND metadata LIKE '%\"stashed\": true%'",
            (self.agent_id,)
        )

        # Group by stash_id
        stashes: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            meta = json.loads(row[0]) if row[0] else {}
            stash_id = meta.get("stash_id")
            if not stash_id:
                continue

            if stash_id not in stashes:
                stashes[stash_id] = {
                    "stash_id": stash_id,
                    "name": meta.get("stash_name", "unnamed"),
                    "message_count": 0,
                    "stashed_at": meta.get("stashed_at")
                }
            stashes[stash_id]["message_count"] += 1

        # Sort by stashed_at descending
        return sorted(
            stashes.values(),
            key=lambda x: x.get("stashed_at", ""),
            reverse=True
        )

    async def get_all_audit_failures(self) -> List[Dict[str, Any]]:
        """
        Retrieves all conversation entries that are marked as audit failures.
        Automatically decrypts content if encryption was enabled.
        """
        rows = await self.db.fetchall(
            "SELECT id, role, content, metadata FROM conversation_history "
            "WHERE agent_id = ? AND deleted_at IS NULL "
            "AND metadata LIKE '%\"audit_failure\": true%'",
            (self.agent_id,)
        )
        results = []
        for row in rows:
            if not row[3]:
                continue
            row_id = row[0]
            meta = json.loads(row[3])
            content, needs_migration = self._decrypt_with_fallback(row[2], meta)

            # Opportunistic migration
            if needs_migration and self._migrate_on_read:
                try:
                    await self._migrate_message(row_id, content)
                except Exception as e:
                    logger.warning(f"Migration failed for message {row_id} in get_all_audit_failures: {e}")

            cleaned_meta = remove_enc_flag(meta)
            if cleaned_meta:
                cleaned_meta.pop('key_version', None)

            results.append({
                "role": row[1],
                "content": content,
                "metadata": cleaned_meta if cleaned_meta else None
            })
        return results
