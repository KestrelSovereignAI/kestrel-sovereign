"""Operator-triggered re-embedding of stored vectors (#2289).

Switching the embedding model/provider strands existing memories:
vectors produced by different models are never comparable (even at
equal dimension), so the ``embedding_profile_id`` filter correctly
hides old rows from kNN and recall silently degrades to keyword
search. This module rewrites the stored vectors of the three embedded
tables (``conversation_history``, ``saved_items``, ``document_chunks``)
into the currently RESOLVED embedding profile so the whole corpus
lives in one comparable coordinate space again.

Design:

- **Idempotent / resumable.** Work is selected by "profile != target
  (or NULL)". A row that has already been re-embedded to the target
  drops out of the working set, so an interrupted run loses nothing —
  re-running simply continues from wherever the last committed batch
  left off. Batch commits go through :class:`AsyncDatabase` so the
  #1701 write-unit serialization is respected (no new locks).

- **Keyset pagination.** Each batch is fetched with ``id > <last id>``
  ordering so the scan always advances even when a row can't be
  embedded (empty source text) and therefore stays "stale" — no
  infinite loop.

- **Dimension guard.** Every produced embedding is length-checked
  against the deployment's vector-column dimension. A mismatch is
  refused at the CLI layer BEFORE any write; the per-row guard here is
  defense-in-depth so a late provider drift can never corrupt the
  column.

The re-embed source text mirrors each table's original writer:

- ``conversation_history`` — decrypted ``content`` (Fernet at rest).
- ``saved_items`` — ``summary`` when present, else ``content[:1000]``
  (matches :meth:`SavedItemsStore.save_item`).
- ``document_chunks`` — plaintext ``content``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .encryption import (
    DecryptionError,
    decrypt_string,
    get_agent_fernet,
    get_fernet,
)
from .sqla.embedding_profile import upsert_embedding_profile

logger = logging.getLogger(__name__)

# Tables that carry an ``embedding_vec`` + ``embedding_profile_id``
# column pair (#1477 / #2289). Order is stable so the CLI reports the
# same table order every run.
REINDEX_TABLES: Tuple[str, ...] = (
    "conversation_history",
    "saved_items",
    "document_chunks",
)


@dataclass
class _TableSpec:
    """Static description of one embedded table for the reindexer."""

    name: str
    id_col: str
    agent_col: Optional[str]
    # Extra columns SELECTed alongside the id so the source text can be
    # rebuilt. Order matters — ``_extract_text`` reads by position.
    text_cols: Tuple[str, ...]


_TABLE_SPECS: Dict[str, _TableSpec] = {
    "conversation_history": _TableSpec(
        name="conversation_history",
        id_col="id",
        agent_col="agent_id",
        # content (ciphertext), metadata (enc flags), agent_id (key scope)
        text_cols=("content", "metadata", "agent_id"),
    ),
    "saved_items": _TableSpec(
        name="saved_items",
        id_col="id",
        agent_col="agent_id",
        text_cols=("summary", "content"),
    ),
    "document_chunks": _TableSpec(
        name="document_chunks",
        id_col="chunk_id",
        agent_col=None,  # global — no agent scoping
        text_cols=("content",),
    ),
}


def _serialize_embedding(embedding: Sequence[float]) -> bytes:
    """Pack an embedding as little-endian float32 bytes for a SQLite BLOB.

    Explicit ``<`` so the bytes round-trip on big-endian hosts — the
    same shape ``async_conversation_store._serialize_embedding`` writes
    and the PurePython vector backend reads.
    """
    return struct.pack(f"<{len(embedding)}f", *embedding)


def _format_pgvector_text(embedding: Sequence[float]) -> str:
    """Format an embedding as pgvector's ``[v1,v2,…]`` bind text."""
    return "[" + ",".join(repr(float(v)) for v in embedding) + "]"


@dataclass
class ReindexStats:
    """Per-table outcome of a reindex pass."""

    table: str
    scanned: int = 0
    reembedded: int = 0
    skipped_empty: int = 0
    skipped_dim_mismatch: int = 0
    failed: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "scanned": self.scanned,
            "reembedded": self.reembedded,
            "skipped_empty": self.skipped_empty,
            "skipped_dim_mismatch": self.skipped_dim_mismatch,
            "failed": self.failed,
        }


async def dominant_embedding_profile(
    db: Any, agent_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Return the corpus's dominant existing embedding profile, or ``None`` (#2366).

    Surveys the ``embedding_profile_id`` stamped on the three embedded tables
    (``conversation_history``, ``saved_items``, ``document_chunks``), tallies
    rows per profile, and returns the human-readable descriptor of the profile
    with the most rows joined from the ``embedding_profiles`` registry:

        {"profile_id", "provider", "model", "dim", "space_id",
         "normalized", "row_count"}

    Auto embedding-model resolution consults this so a fresh default prefers a
    model that keeps new memories in the same coordinate space old memories
    already live in (continuity beats catalog order). Best-effort: any read
    failure, an empty corpus, or a dominant profile with no registry row yields
    ``None`` and the caller falls back to hint/catalog order.
    """
    counts: Dict[str, int] = {}
    for table, spec in _TABLE_SPECS.items():
        where = "embedding_profile_id IS NOT NULL"
        params: List[Any] = []
        if spec.agent_col and agent_id:
            where += f" AND {spec.agent_col} = ?"
            params.append(agent_id)
        try:
            rows = await db.fetchall(
                f"SELECT embedding_profile_id, COUNT(*) FROM {spec.name} "
                f"WHERE {where} GROUP BY embedding_profile_id",
                tuple(params),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("dominant profile scan of %s failed: %s", table, exc)
            continue
        for row in rows or []:
            pid, n = row[0], row[1]
            if not pid:
                continue
            counts[pid] = counts.get(pid, 0) + int(n or 0)

    if not counts:
        return None
    dominant_id = max(counts, key=lambda pid: counts[pid])

    try:
        prof_rows = await db.fetchall(
            "SELECT provider, model, dim, space_id, normalized "
            "FROM embedding_profiles WHERE id = ?",
            (dominant_id,),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("embedding_profiles lookup failed for %s: %s", dominant_id, exc)
        return None
    if not prof_rows or not prof_rows[0]:
        # Rows exist under this profile id but the registry has no descriptor
        # (pre-registry write). Nothing to match a discovered model against.
        return None

    provider, model, dim, space_id, normalized = prof_rows[0]
    return {
        "profile_id": dominant_id,
        "provider": provider,
        "model": model,
        "dim": int(dim) if dim is not None else None,
        "space_id": space_id,
        "normalized": bool(normalized),
        "row_count": counts[dominant_id],
    }


class EmbeddingReindexer:
    """Rewrite stored vectors into the resolved embedding profile.

    Args:
        db: The :class:`AsyncDatabase` to sweep.
        embedding_service: An object exposing ``aembed_batch(texts) ->
            list[Optional[list[float]]]`` and ``describe()`` (the
            :class:`ProviderEmbeddingService` shape). Used to re-embed
            source text and to register the target profile.
        target_profile_id: The 12-char profile id every rewritten row
            is stamped with.
        column_dim: The width of the ``embedding_vec`` column. Produced
            embeddings whose length differs are skipped (never written)
            to avoid corrupting the column. When ``None`` the guard is
            disabled (all lengths accepted).
        batch_size: Rows re-embedded per batch (and per commit sweep).
        rate_limit_s: Seconds to sleep between batches — throttle for
            rate-limited cloud embedding providers. ``0`` disables.
    """

    def __init__(
        self,
        db: Any,
        embedding_service: Any,
        target_profile_id: str,
        *,
        column_dim: Optional[int] = None,
        batch_size: int = 500,
        rate_limit_s: float = 0.0,
    ) -> None:
        if not target_profile_id:
            raise ValueError("target_profile_id is required")
        self.db = db
        self.service = embedding_service
        self.target = target_profile_id
        self.column_dim = int(column_dim) if column_dim else None
        self.batch_size = max(1, int(batch_size))
        self.rate_limit_s = max(0.0, float(rate_limit_s))
        self.backend_type = getattr(db, "backend_type", None)
        self._fernet_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------ stale

    def _stale_where(self, spec: _TableSpec, agent_id: Optional[str]) -> Tuple[str, List[Any]]:
        """Return the ``WHERE`` predicate + params for rows not on target.

        A row is stale when it carries no vector at all (``embedding_vec
        IS NULL`` — never embedded, or dropped by a partial migration /
        fallback path), when its ``embedding_profile_id`` is NULL (never
        stamped / pre-#1477), or when the profile differs from the
        target. The ``embedding_vec IS NULL`` arm matters even for rows
        already stamped with the target profile: a partial migration can
        leave ``embedding_profile_id = target`` with a NULL vector, and
        without this those rows stay invisible to kNN forever (#2289).
        """
        clause = (
            "(embedding_vec IS NULL OR embedding_profile_id IS NULL "
            "OR embedding_profile_id <> ?)"
        )
        params: List[Any] = [self.target]
        if spec.agent_col and agent_id:
            clause += f" AND {spec.agent_col} = ?"
            params.append(agent_id)
        return clause, params

    async def count_stale(self, table: str, agent_id: Optional[str] = None) -> int:
        """Count rows in *table* whose profile differs from the target."""
        spec = _TABLE_SPECS[table]
        where, params = self._stale_where(spec, agent_id)
        rows = await self.db.fetchall(
            f"SELECT COUNT(*) FROM {spec.name} WHERE {where}", tuple(params)
        )
        return int(rows[0][0]) if rows and rows[0] else 0

    async def count_all_stale(
        self, agent_id: Optional[str] = None, tables: Optional[Sequence[str]] = None
    ) -> Dict[str, int]:
        """Count stale rows for each table (skips missing tables/columns)."""
        out: Dict[str, int] = {}
        for table in tables or REINDEX_TABLES:
            try:
                out[table] = await self.count_stale(table, agent_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("count_stale(%s) failed: %s", table, exc)
                out[table] = 0
        return out

    # --------------------------------------------------------------- decryption

    def _fernet_for(self, agent_id: Optional[str]) -> Any:
        """Per-agent Fernet, cached. Falls back to the global key."""
        key = agent_id or ""
        if key not in self._fernet_cache:
            self._fernet_cache[key] = get_agent_fernet(agent_id) if agent_id else None
        return self._fernet_cache[key]

    def _decrypt_content(
        self, content: Optional[str], metadata_json: Optional[str], agent_id: Optional[str]
    ) -> Optional[str]:
        """Decrypt an encrypted ``content`` column, tolerant of plaintext.

        Rows written without a data key are stored as plaintext with no
        ``enc`` flag — returned unchanged. Undecryptable rows (wrong
        key) are skipped (``None``) rather than crashing the sweep.
        """
        if content is None:
            return None
        meta: Optional[Dict[str, Any]] = None
        if metadata_json:
            try:
                meta = json.loads(metadata_json)
            except (json.JSONDecodeError, TypeError):
                meta = None
        if not meta or not meta.get("enc"):
            return content
        for fernet in (self._fernet_for(agent_id), get_fernet()):
            if fernet is None:
                continue
            try:
                return decrypt_string(content, meta, fernet)
            except DecryptionError:
                continue
        logger.warning(
            "Could not decrypt conversation_history row for agent %s; "
            "skipping it during reindex.",
            agent_id,
        )
        return None

    def _extract_text(self, spec: _TableSpec, values: Tuple[Any, ...]) -> Optional[str]:
        """Rebuild the embed-source text for a row from its SELECTed cols."""
        if spec.name == "conversation_history":
            content, metadata_json, agent_id = values
            return self._decrypt_content(content, metadata_json, agent_id)
        if spec.name == "saved_items":
            summary, content = values
            if summary:
                return summary
            return content[:1000] if content else None
        # document_chunks
        (content,) = values
        return content

    # ------------------------------------------------------------------- write

    async def _write_row(self, spec: _TableSpec, row_id: Any, embedding: List[float]) -> None:
        """Stamp a single row with the new vector + target profile id."""
        if self.backend_type == "postgres":
            await self.db.execute_commit(
                f"UPDATE {spec.name} SET embedding_vec = ?::vector, "
                f"embedding_profile_id = ? WHERE {spec.id_col} = ?",
                (_format_pgvector_text(embedding), self.target, row_id),
            )
        else:
            await self.db.execute_commit(
                f"UPDATE {spec.name} SET embedding_vec = ?, "
                f"embedding_profile_id = ? WHERE {spec.id_col} = ?",
                (_serialize_embedding(embedding), self.target, row_id),
            )

    # -------------------------------------------------------------------- main

    async def reindex_table(
        self,
        table: str,
        agent_id: Optional[str] = None,
        *,
        progress: Optional[Callable[[ReindexStats], None]] = None,
    ) -> ReindexStats:
        """Re-embed every stale row of *table* into the target profile."""
        spec = _TABLE_SPECS[table]
        stats = ReindexStats(table=table)
        after: Optional[Any] = None
        select_cols = ", ".join((spec.id_col, *spec.text_cols))

        while True:
            where, params = self._stale_where(spec, agent_id)
            page_params = list(params)
            if after is not None:
                where += f" AND {spec.id_col} > ?"
                page_params.append(after)
            sql = (
                f"SELECT {select_cols} FROM {spec.name} WHERE {where} "
                f"ORDER BY {spec.id_col} LIMIT ?"
            )
            page_params.append(self.batch_size)
            rows = await self.db.fetchall(sql, tuple(page_params))
            if not rows:
                break

            # Advance the keyset cursor before any per-row filtering so
            # un-embeddable rows (empty text) can't stall the scan.
            after = rows[-1][0]
            stats.scanned += len(rows)

            pending: List[Tuple[Any, str]] = []
            for row in rows:
                row_id = row[0]
                text = self._extract_text(spec, tuple(row[1:]))
                if not text or not text.strip():
                    stats.skipped_empty += 1
                    continue
                pending.append((row_id, text))

            if pending:
                batch_failed = False
                try:
                    embeddings = await self.service.aembed_batch(
                        [t for _, t in pending]
                    )
                except Exception as exc:
                    logger.warning(
                        "aembed_batch failed for %s batch (%s); skipping batch.",
                        table, exc,
                    )
                    batch_failed = True
                    embeddings = [None] * len(pending)

                empty_in_batch = 0
                for (row_id, _text), embedding in zip(pending, embeddings):
                    if not embedding:
                        stats.failed += 1
                        empty_in_batch += 1
                        continue
                    if self.column_dim is not None and len(embedding) != self.column_dim:
                        stats.skipped_dim_mismatch += 1
                        continue
                    try:
                        await self._write_row(spec, row_id, list(embedding))
                        stats.reembedded += 1
                    except Exception as exc:
                        logger.warning(
                            "Failed to write reindexed vector for %s row %s: %s",
                            table, row_id, exc,
                        )
                        stats.failed += 1

                # A dead / mis-resolved embedding service returns empty vectors
                # for every row with no exception — the exact silent
                # scanned-N/reembedded-0 failure of #2360. Make it loud: without
                # this log an operator sees only "0 re-embedded, error: null".
                if empty_in_batch and not batch_failed:
                    logger.warning(
                        "reindex %s: embedding service returned empty vectors "
                        "for %d/%d rows in this batch (nothing written). The "
                        "resolved embedding service is likely dead or "
                        "mis-configured — check the active embedding route.",
                        table, empty_in_batch, len(pending),
                    )

            if progress is not None:
                progress(stats)
            if self.rate_limit_s:
                await asyncio.sleep(self.rate_limit_s)

        # Best-effort: register the target profile so ``embeddings
        # audit`` can map its id → human-readable provider/model/dim.
        if stats.reembedded:
            try:
                await upsert_embedding_profile(self.db, self.service, self.target)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("target profile registry upsert failed: %s", exc)

        return stats
