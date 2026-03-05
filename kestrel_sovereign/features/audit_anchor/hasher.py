"""
Deterministic hashing of audit log entries.

Provides the cryptographic foundation for audit trail anchoring.
Same entries always produce the same SHA-256 hash regardless of
insertion order, enabling tamper detection (Article II Right 3).
"""

import hashlib
import json
from typing import List, Dict


class AuditHasher:
    """Creates deterministic hashes of audit log entries."""

    @staticmethod
    def hash_entries(entries: List[Dict]) -> str:
        """
        Hash entries deterministically. Same entries always produce same hash.

        Args:
            entries: List of audit log entry dicts

        Returns:
            SHA-256 hex digest of the serialized entries
        """
        serialized = AuditHasher.serialize_entries(entries)
        return hashlib.sha256(serialized).hexdigest()

    @staticmethod
    def serialize_entries(entries: List[Dict]) -> bytes:
        """
        Serialize entries deterministically for storage.

        Entries are sorted by created_at (falling back to id) so that
        the same set of entries always serializes identically regardless
        of the order they were retrieved in.

        Args:
            entries: List of audit log entry dicts

        Returns:
            UTF-8 encoded JSON bytes with sorted keys and compact separators
        """
        # Sort entries by timestamp (or id as fallback)
        sorted_entries = sorted(
            entries,
            key=lambda e: e.get("created_at", e.get("timestamp", e.get("id", "")))
        )
        # Use sorted keys and consistent separators for determinism
        return json.dumps(
            sorted_entries, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
