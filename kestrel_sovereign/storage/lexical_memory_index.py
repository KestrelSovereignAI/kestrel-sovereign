"""Privacy-preserving exact-token candidate index for conversation memory.

Conversation bodies may be Fernet-encrypted at rest, so ordinary SQL/FTS
cannot shortlist lexical matches.  The previous correctness bridge decrypted
every row outside the active embedding profile on every recall.  This module
stores keyed HMAC token digests instead: the database can match equality and
rank overlap without learning the plaintext tokens.

The index is deliberately a *candidate* index.  Callers still decrypt and run
the canonical tokenizer before returning a memory.  False positives therefore
cannot change recall semantics, while HMAC collisions are cryptographically
negligible.  Rows not covered by the current key version remain eligible for
the legacy scan, so an interrupted backfill or key rotation never makes an old
fact unreachable.
"""

from __future__ import annotations

import hashlib
import hmac
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, List, Optional, Sequence
from uuid import uuid4

from kestrel_sovereign.security.encryption import (
    MasterKeyNotConfiguredError,
    get_agent_key,
)


_DOMAIN = b"kestrel:conversation-lexical-index:v1\0"
_CLEANUP_LOCK_DOMAIN = b"kestrel:conversation-lexical-cleanup-lock:v1\0"
MAX_INDEXED_QUERY_TOKENS = 100


def _cleanup_lock_id(agent_id: str, lexical_index_id: str) -> int:
    """Return a stable signed-64-bit PostgreSQL advisory-lock key.

    Hash collisions only serialize unrelated cleanup work; they cannot weaken
    correctness. Length-prefixing keeps distinct ``(agent, key)`` pairs from
    sharing an input representation before hashing.
    """
    agent_bytes = agent_id.encode("utf-8")
    key_bytes = lexical_index_id.encode("utf-8")
    payload = b"".join(
        (
            _CLEANUP_LOCK_DOMAIN,
            len(agent_bytes).to_bytes(4, "big"),
            agent_bytes,
            len(key_bytes).to_bytes(4, "big"),
            key_bytes,
        )
    )
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
        signed=True,
    )


@dataclass(frozen=True)
class LexicalIndexHealth:
    """Durable per-agent coverage of the current blind-index key."""

    total_live: int
    indexed_current: int
    unindexed: int
    coverage: float
    index_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_live": self.total_live,
            "indexed_current": self.indexed_current,
            "unindexed": self.unindexed,
            "coverage": self.coverage,
            "index_version": self.index_version,
        }


@dataclass(frozen=True)
class LexicalIndexReplacement:
    """One optimistic, atomic backfill replacement for an existing owner."""

    message_id: int
    expected_key: Optional[str]
    replacement_key: str
    tokens: Iterable[str]


@dataclass(frozen=True)
class LexicalIndexReplacementResult:
    """Durable effects of one atomic replacement batch."""

    updated: int
    garbage_collected: int


class ConversationLexicalIndex:
    """Keyed token index scoped to one agent."""

    def __init__(self, db: Any, agent_id: str) -> None:
        self.db = db
        self.agent_id = agent_id
        self._key, self.version = self._derive_key(agent_id)

    @staticmethod
    def _derive_key(agent_id: str) -> tuple[bytes, str]:
        if not agent_id:
            # Legacy unscoped stores are plaintext-only: encryption correctly
            # rejects an empty DID, but the compatibility store must still be
            # constructible for migration and delegation paths.
            key = hashlib.sha256(_DOMAIN + b"legacy-unscoped").digest()
            fingerprint = hashlib.sha256(_DOMAIN + key).hexdigest()[:16]
            return key, f"v1:plaintext:{fingerprint}"
        try:
            conversation_key = get_agent_key(agent_id, "conversations")
            key = hmac.new(conversation_key, _DOMAIN, hashlib.sha256).digest()
            mode = "keyed"
        except MasterKeyNotConfiguredError:
            # A plaintext deployment has no confidentiality boundary to widen.
            # Keep its index deterministic so enabling encryption later changes
            # the version and safely routes old rows through backfill/fallback.
            key = hashlib.sha256(_DOMAIN + agent_id.encode("utf-8")).digest()
            mode = "plaintext"
        fingerprint = hashlib.sha256(_DOMAIN + key).hexdigest()[:16]
        return key, f"v1:{mode}:{fingerprint}"

    def hash_token(self, token: str) -> str:
        return hmac.new(
            self._key,
            _DOMAIN + token.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def backfill_message_key(self, row_id: int) -> str:
        """Stable opaque key so interrupted backfills resume idempotently."""
        return hmac.new(
            self._key,
            _DOMAIN + f"message-key:{int(row_id)}".encode("ascii"),
            hashlib.sha256,
        ).hexdigest()[:32]

    def token_hashes(self, tokens: Iterable[str]) -> List[str]:
        return [self.hash_token(token) for token in dict.fromkeys(tokens)]

    async def index_message(self, message_key: str, tokens: Iterable[str]) -> None:
        """Replace one message's digest set before its coverage marker commits."""
        await self.index_messages([(message_key, tokens)])

    async def index_messages(
        self, entries: Sequence[tuple[str, Iterable[str]]]
    ) -> None:
        """Batch replacement used by large-corpus resumable backfills."""
        if not entries:
            return
        await self.db.execute_many(
            "DELETE FROM conversation_lexical_tokens "
            "WHERE agent_id = ? AND lexical_index_id = ?",
            [(self.agent_id, message_key) for message_key, _tokens in entries],
        )
        token_rows = []
        for message_key, tokens in entries:
            hashes = self.token_hashes(tokens)
            token_rows.extend(
                (self.agent_id, message_key, digest) for digest in hashes
            )
        if not token_rows:
            return
        await self.db.execute_many(
            "INSERT INTO conversation_lexical_tokens "
            "(agent_id, lexical_index_id, token_hash) VALUES (?, ?, ?) "
            "ON CONFLICT (agent_id, lexical_index_id, token_hash) DO NOTHING",
            token_rows,
        )

    @asynccontextmanager
    async def serialized_token_cleanup(
        self,
        lexical_index_ids: Iterable[str],
    ) -> AsyncIterator[tuple[str, ...]]:
        """Serialize owner checks and token deletion for exact key sets.

        PostgreSQL MVCC lets two transactions deleting the final two owners of
        one shared key each see the other's uncommitted owner. Without a key
        boundary, both ``NOT EXISTS`` checks can therefore retain the token
        set. Transaction-scoped advisory locks close that gap independently of
        token-row existence. SQLite's single writer is reserved explicitly.

        Lock IDs are globally sorted before acquisition so overlapping
        multi-key cleanup sets cannot introduce an advisory-lock order cycle.
        The surrounding transaction (opened here or reused by nesting) owns
        the locks through the caller's owner check and deletion. Backfill also
        takes the boundary before replacement-token writes, preserving one
        global order: history rows, advisory keys, then token rows.
        """
        ordered_keys = tuple(
            sorted(
                {
                    str(key)
                    for key in lexical_index_ids
                    if key is not None
                }
            )
        )
        if not ordered_keys:
            yield ordered_keys
            return

        async with self.db.transaction():
            if self.db.backend_type == "postgres":
                lock_ids = sorted(
                    {_cleanup_lock_id(self.agent_id, key) for key in ordered_keys}
                )
                for lock_id in lock_ids:
                    await self.db.fetchval(
                        "SELECT pg_advisory_xact_lock(?)",
                        (lock_id,),
                    )
            elif self.db.backend_type == "sqlite":
                # SQLite transactions begin deferred. Promote an otherwise
                # standalone cleanup scope to the one serialized writer slot.
                await self.db.execute(
                    "DELETE FROM conversation_lexical_tokens WHERE 0"
                )
            else:  # pragma: no cover - AsyncDatabase exposes only these two
                raise RuntimeError(
                    "Token cleanup serialization does not support backend "
                    f"{self.db.backend_type!r}"
                )
            yield ordered_keys

    async def _has_other_owner(
        self,
        lexical_index_id: str,
        message_id: int,
    ) -> bool:
        """Whether any current, trashed, or archived row shares this key."""
        owner = await self.db.fetchval(
            "SELECT 1 FROM conversation_history "
            "WHERE agent_id = ? AND lexical_index_id = ? "
            "AND id != ? LIMIT 1",
            (self.agent_id, lexical_index_id, message_id),
        )
        return owner is not None

    async def replace_existing_messages(
        self,
        entries: Sequence[LexicalIndexReplacement],
    ) -> LexicalIndexReplacementResult:
        """Atomically replace backfill tokens and coverage for live rows.

        Hydration necessarily happens before this method.  Re-checking and
        updating each owner inside the same transaction as its token writes
        prevents a stale backfill from recreating blind-index residue after a
        concurrent hard purge.  Cancellation or any write failure rolls both
        the coverage marker and token set back together. If legacy data already
        assigns the deterministic replacement key to another row, this method
        moves the target to a fresh opaque key rather than clobbering the shared
        digest set.
        """
        if not entries:
            return LexicalIndexReplacementResult(updated=0, garbage_collected=0)

        updated = 0
        garbage_collected = 0
        updated_entries: list[tuple[LexicalIndexReplacement, str]] = []
        owner_key_predicate = (
            "lexical_index_id IS NOT DISTINCT FROM ?"
            if self.db.backend_type == "postgres"
            else "lexical_index_id IS ?"
        )
        # Match the hard-purge row-lock order before taking any advisory key
        # locks. Direct callers are not required to pre-sort their batch.
        prepared_entries = [
            (entry, uuid4().hex)
            for entry in sorted(entries, key=lambda candidate: candidate.message_id)
        ]
        async with self.db.transaction():
            for entry, collision_fallback in prepared_entries:
                affected = await self.db.execute(
                    "UPDATE conversation_history "
                    "SET lexical_index_id = ?, lexical_index_version = ? "
                    "WHERE agent_id = ? AND id = ? "
                    "AND deleted_at IS NULL AND archived_at IS NULL "
                    f"AND {owner_key_predicate}",
                    (
                        entry.replacement_key,
                        self.version,
                        self.agent_id,
                        entry.message_id,
                        entry.expected_key,
                    ),
                )
                if not affected:
                    continue

                updated += 1
                updated_entries.append((entry, collision_fallback))

            # Lock every possible old/final key before resolving replacement
            # ownership.  The fresh fallback is pre-generated so collision
            # handling never acquires a late advisory lock out of global order.
            token_mutation_keys: set[str] = set()
            for entry, collision_fallback in updated_entries:
                if entry.expected_key is not None:
                    token_mutation_keys.add(entry.expected_key)
                token_mutation_keys.add(entry.replacement_key)
                token_mutation_keys.add(collision_fallback)

            async with self.serialized_token_cleanup(
                token_mutation_keys
            ) as cleanup_keys:
                resolved_entries: list[LexicalIndexReplacement] = []
                obsolete_keys: set[str] = set()
                for entry, collision_fallback in updated_entries:
                    replacement_key = entry.replacement_key
                    other_owner = await self._has_other_owner(
                        replacement_key,
                        entry.message_id,
                    )
                    if other_owner:
                        replacement_key = collision_fallback
                        fallback_owner = await self._has_other_owner(
                            replacement_key,
                            entry.message_id,
                        )
                        if fallback_owner:
                            raise RuntimeError(
                                "Fresh lexical replacement key is already owned; "
                                "retry the atomic backfill batch"
                            )
                        moved = await self.db.execute(
                            "UPDATE conversation_history "
                            "SET lexical_index_id = ? "
                            "WHERE agent_id = ? AND id = ? "
                            "AND lexical_index_id = ?",
                            (
                                replacement_key,
                                self.agent_id,
                                entry.message_id,
                                entry.replacement_key,
                            ),
                        )
                        if moved != 1:
                            raise RuntimeError(
                                "Lexical replacement owner changed during "
                                "collision resolution"
                            )
                        # The proposed key was assigned only tentatively.  If
                        # no survivor owns it, reclaim any pre-existing residue.
                        obsolete_keys.add(entry.replacement_key)

                    # Recheck immediately before token mutation.  All Kestrel
                    # backfill/purge writers for this key are now serialized;
                    # a remaining owner means replacing the digest set would
                    # silently remove that row from exact lexical recall.
                    if await self._has_other_owner(
                        replacement_key,
                        entry.message_id,
                    ):
                        raise RuntimeError(
                            "Lexical replacement key remains multiply owned"
                        )

                    resolved_entry = LexicalIndexReplacement(
                        message_id=entry.message_id,
                        expected_key=entry.expected_key,
                        replacement_key=replacement_key,
                        tokens=entry.tokens,
                    )
                    resolved_entries.append(resolved_entry)
                    if (
                        entry.expected_key is not None
                        and entry.expected_key != replacement_key
                    ):
                        obsolete_keys.add(entry.expected_key)

                for entry in resolved_entries:
                    await self.db.execute(
                        "DELETE FROM conversation_lexical_tokens "
                        "WHERE agent_id = ? AND lexical_index_id = ?",
                        (self.agent_id, entry.replacement_key),
                    )
                    token_rows = [
                        (self.agent_id, entry.replacement_key, digest)
                        for digest in self.token_hashes(entry.tokens)
                    ]
                    if token_rows:
                        await self.db.execute_many(
                            "INSERT INTO conversation_lexical_tokens "
                            "(agent_id, lexical_index_id, token_hash) "
                            "VALUES (?, ?, ?) "
                            "ON CONFLICT "
                            "(agent_id, lexical_index_id, token_hash) "
                            "DO NOTHING",
                            token_rows,
                        )

                for old_key in cleanup_keys:
                    if old_key not in obsolete_keys:
                        continue
                    garbage_collected += await self.db.execute(
                        "DELETE FROM conversation_lexical_tokens "
                        "WHERE agent_id = ? AND lexical_index_id = ? "
                        "AND NOT EXISTS ("
                        "SELECT 1 FROM conversation_history "
                        "WHERE agent_id = ? "
                        "AND conversation_history.lexical_index_id = "
                        "conversation_lexical_tokens.lexical_index_id)",
                        (self.agent_id, old_key, self.agent_id),
                    )

        return LexicalIndexReplacementResult(
            updated=updated,
            garbage_collected=garbage_collected,
        )

    async def candidate_message_ids(
        self,
        tokens: Sequence[str],
        *,
        limit: int,
        excluded_embedding_profile_id: Optional[str] = None,
    ) -> List[int]:
        """Return ids ranked by matched-token count then recency.

        Queries beyond ``MAX_INDEXED_QUERY_TOKENS`` deliberately use the
        complete fallback scan.  Chunking and independently limiting partial
        rankings can silently omit an older row with the best *global* token
        overlap; the bounded common path must not weaken exact recall.
        """
        hashes = self.token_hashes(tokens)
        if not hashes:
            return []
        if len(hashes) > MAX_INDEXED_QUERY_TOKENS:
            raise ValueError(
                f"query has {len(hashes)} searchable tokens; blind-index "
                f"budget is {MAX_INDEXED_QUERY_TOKENS}"
            )
        fetch_limit = max(1, int(limit))
        placeholders = ",".join("?" for _ in hashes)
        profile_clause = ""
        params: list[Any] = [self.agent_id, self.version, *hashes]
        if excluded_embedding_profile_id is not None:
            profile_clause = (
                " AND (c.embedding_profile_id IS NULL "
                "OR c.embedding_profile_id != ?)"
            )
            params.append(excluded_embedding_profile_id)
        params.append(fetch_limit)
        rows = await self.db.fetchall(
            "SELECT c.id, COUNT(t.token_hash) AS matched "
            # CROSS JOIN preserves the selective token-first loop on SQLite.
            # A regular JOIN made its planner walk every live conversation via
            # the archived-row index before probing the digest table (190 ms
            # at 100k rows). PostgreSQL remains free to optimize this inner
            # join normally.
            "FROM conversation_lexical_tokens t "
            "CROSS JOIN conversation_history c "
            "WHERE t.agent_id = ? AND c.agent_id = t.agent_id "
            "AND c.lexical_index_id = t.lexical_index_id "
            "AND c.lexical_index_version = ? "
            f"AND t.token_hash IN ({placeholders}) "
            "AND c.deleted_at IS NULL AND c.archived_at IS NULL"
            f"{profile_clause} GROUP BY c.id "
            "ORDER BY matched DESC, c.id DESC LIMIT ?",
            tuple(params),
        )
        return [int(row[0]) for row in rows]

    def coverage_gap_sql(self, alias: str = "conversation_history") -> tuple[str, tuple[Any, ...]]:
        """Predicate selecting rows not durably indexed by this key version."""
        # Writers persist the token set *before* committing this marker.  The
        # marker is therefore a durable completion record, not an optimistic
        # intent flag.  Keeping this predicate marker-only lets its composite
        # index prove complete coverage without a correlated 100k-row token
        # table walk on every recall.
        predicate = (
            f"({alias}.lexical_index_version IS NULL "
            f"OR {alias}.lexical_index_version < ? "
            f"OR {alias}.lexical_index_version > ? "
            f"OR ({alias}.lexical_index_version = ? "
            f"AND {alias}.lexical_index_id IS NULL))"
        )
        return predicate, (self.version, self.version, self.version)

    def coverage_gap_predicates(
        self, alias: str = "conversation_history"
    ) -> tuple[tuple[str, tuple[Any, ...]], ...]:
        """Disjoint, indexable forms of :meth:`coverage_gap_sql`.

        SQLite often chooses the live-row index for the OR predicate and
        walks the whole corpus just to prove coverage is complete.  Separate
        range probes use the composite coverage index and make recall cost
        proportional to uncovered rows instead.
        """
        version = f"{alias}.lexical_index_version"
        return (
            (f"{version} IS NULL", ()),
            (f"{version} < ?", (self.version,)),
            (f"{version} > ?", (self.version,)),
            (
                f"{version} = ? AND {alias}.lexical_index_id IS NULL",
                (self.version,),
            ),
        )

    async def health(self) -> LexicalIndexHealth:
        gap, gap_params = self.coverage_gap_sql("c")
        row = await self.db.fetchone(
            "SELECT COUNT(*), "
            f"SUM(CASE WHEN {gap} THEN 0 ELSE 1 END) "
            "FROM conversation_history c WHERE c.agent_id = ? "
            "AND c.deleted_at IS NULL AND c.archived_at IS NULL",
            (*gap_params, self.agent_id),
        )
        total = int(row[0] or 0) if row else 0
        indexed = int(row[1] or 0) if row else 0
        unindexed = max(0, total - indexed)
        return LexicalIndexHealth(
            total_live=total,
            indexed_current=indexed,
            unindexed=unindexed,
            coverage=(indexed / total if total else 1.0),
            index_version=self.version,
        )
