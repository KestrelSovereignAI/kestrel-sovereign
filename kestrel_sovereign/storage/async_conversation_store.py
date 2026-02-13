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
from .encryption import (
    get_fernet, get_agent_fernet, encrypt_string, decrypt_string, remove_enc_flag,
    DecryptionError
)

logger = logging.getLogger(__name__)

# Current key version for new encryptions
CURRENT_KEY_VERSION = 1


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
        """
        meta = dict(metadata) if metadata else {}

        # Store session_id in metadata for session continuity
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
            # Default behavior: get most recent messages
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, created_at FROM conversation_history "
                "WHERE agent_id = ? ORDER BY id DESC LIMIT ?",
                (self.agent_id, limit)
            )
        history = []
        for row in reversed(rows):  # Return in chronological order
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else None

            content, needs_migration = self._decrypt_with_fallback(row[2], meta)

            # Opportunistic migration to per-agent key
            if needs_migration and self._migrate_on_read:
                try:
                    await self._migrate_message(row_id, content)
                except Exception as e:
                    logger.warning(f"Migration failed for message {row_id}: {e}")

            entry = {
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

    async def _get_session_messages(self, session_id: str, limit: int) -> List[tuple]:
        """Get messages belonging to a specific session.

        Sessions are determined by:
        1. Time-based grouping (30-minute gaps end a session)
        2. Explicit session_id in metadata (for resumed conversations)

        The session_id is the message ID that starts that session.

        Args:
            session_id: The message ID that marks the session start
            limit: Maximum messages to return

        Returns:
            List of raw rows (id, role, content, metadata, created_at)
        """
        from datetime import datetime

        SESSION_GAP_MINUTES = 30

        # Get the start time of this session (if session_id is a message ID)
        start_row = await self.db.fetchone(
            "SELECT created_at FROM conversation_history WHERE id = ? AND agent_id = ?",
            (session_id, self.agent_id)
        )

        # If session_id is a message ID, get messages from that timestamp forward
        all_rows = []
        if start_row:
            start_time = start_row[0]
            all_rows = await self.db.fetchall(
                """SELECT id, role, content, metadata, created_at
                   FROM conversation_history
                   WHERE agent_id = ? AND created_at >= ?
                   ORDER BY created_at ASC
                   LIMIT ?""",
                (self.agent_id, start_time, limit * 2)  # Fetch extra in case of filtering
            )

        # Also get messages that explicitly belong to this session (resumed conversations)
        # These are messages with session_id in metadata that may come after a time gap
        resumed_rows = await self.db.fetchall(
            """SELECT id, role, content, metadata, created_at
               FROM conversation_history
               WHERE agent_id = ? AND metadata LIKE ?
               ORDER BY created_at ASC
               LIMIT ?""",
            (self.agent_id, f'%"session_id": "{session_id}"%', limit)
        )

        # Also try without space after colon (JSON formatting varies)
        resumed_rows_alt = await self.db.fetchall(
            """SELECT id, role, content, metadata, created_at
               FROM conversation_history
               WHERE agent_id = ? AND metadata LIKE ?
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
        """Get complete conversation history with automatic decryption."""
        rows = await self.db.fetchall(
            "SELECT id, role, content, metadata FROM conversation_history "
            "WHERE agent_id = ? ORDER BY id ASC",
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

    async def search_history(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search conversation history.

        Fetches and decrypts messages, then filters client-side.
        This approach works correctly with encrypted storage.
        """
        # Fetch all messages (up to 5000) and search client-side after decryption
        # SQL LIKE doesn't work on encrypted content, so we must decrypt first
        rows = await self.db.fetchall(
            "SELECT id, role, content, metadata FROM conversation_history "
            "WHERE agent_id = ? ORDER BY id DESC LIMIT 5000",
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
        """Clear conversation history for this agent."""
        await self.db.execute_commit(
            "DELETE FROM conversation_history WHERE agent_id = ?",
            (self.agent_id,)
        )

    async def delete_message(self, message_id: int) -> bool:
        """Delete a specific message by ID.

        Args:
            message_id: The database ID of the message to delete

        Returns:
            True if deleted, False if not found
        """
        result = await self.db.execute_commit(
            "DELETE FROM conversation_history WHERE id = ? AND agent_id = ?",
            (message_id, self.agent_id)
        )
        return result.rowcount > 0 if hasattr(result, 'rowcount') else True

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
        include_stashed: bool = False
    ) -> List[Dict[str, Any]]:
        """Get complete conversation history with message IDs.

        Args:
            include_excluded: If True, include messages marked as excluded from context
            include_stashed: If True, include messages that are stashed

        Returns:
            List of message dicts with 'id', 'role', 'content', 'metadata', 'created_at'
        """
        rows = await self.db.fetchall(
            "SELECT id, role, content, metadata, created_at FROM conversation_history "
            "WHERE agent_id = ? ORDER BY id ASC",
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
                'created_at': row[4]
            }
            history.append(entry)
        return history

    async def update_message_metadata(
        self,
        message_id: int,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """Update metadata for a specific message.

        Args:
            message_id: The message ID to update
            metadata_updates: Dict of metadata fields to update (merged with existing)

        Returns:
            True if message was found and updated, False otherwise
        """
        # Get current metadata
        row = await self.db.fetchone(
            "SELECT metadata FROM conversation_history WHERE id = ? AND agent_id = ?",
            (message_id, self.agent_id)
        )
        if not row:
            logger.warning(f"Message {message_id} not found for agent {self.agent_id}")
            return False

        # Merge with existing metadata
        current_meta = json.loads(row[0]) if row[0] else {}
        current_meta.update(metadata_updates)

        # Update in database
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
            f"WHERE id IN ({placeholders}) AND agent_id = ? ORDER BY id ASC",
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
            "WHERE agent_id = ? AND metadata LIKE '%\"excluded_from_context\": true%' "
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
                "WHERE agent_id = ? AND metadata LIKE ? "
                "ORDER BY id ASC LIMIT ?",
                (self.agent_id, f'%"stash_id": "{stash_id}"%', limit)
            )
        else:
            # Get all stashed messages
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, created_at FROM conversation_history "
                "WHERE agent_id = ? AND metadata LIKE '%\"stashed\": true%' "
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
            "WHERE agent_id = ? AND metadata LIKE '%\"stashed\": true%'",
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
            "WHERE agent_id = ? AND metadata LIKE '%\"audit_failure\": true%'",
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
