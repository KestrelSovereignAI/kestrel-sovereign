"""SQLAlchemy mapping for the ``embedding_profiles`` registry (#1477).

This tiny table records every ``(provider, model, dim, space_id,
normalized)`` combination that has ever produced an embedding in this
deployment. The 12-char ``id`` is derived deterministically from
those fields (see
:func:`kestrel_sovereign.llm.embedding_service.derive_embedding_profile`)
so the same configuration always lands on the same row regardless of
restart.

The registry is purely descriptive — it doesn't gate writes. Storage
code stamps the derived id directly onto the three embedded tables
(``conversation_history``, ``saved_items``, ``document_chunks``) at
write time, and the kNN profile filter reads that stamp without
joining the registry. The registry exists so operators can answer
"what profiles do I have rows from?" without parsing 8M
``embedding_profile_id`` values. The ``audit`` CLI subcommand reads
it.

Writes are upsert-by-id; reads are a single ``SELECT * FROM
embedding_profiles``. Tiny by design.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional, Set

from sqlalchemy import Boolean, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import SovereignBase

logger = logging.getLogger(__name__)


# Per-process cache of profile ids we've already upserted, so the
# storage hot path doesn't do an upsert + roundtrip on every write.
# Tiny by design — there are typically 1-2 distinct profiles per
# deployment, rarely more than ~5 across mid-life provider switches.
_PROFILE_UPSERT_CACHE: Set[str] = set()


def _clear_profile_upsert_cache_for_tests() -> None:
    """Reset the process-local upsert cache. Test-only."""
    _PROFILE_UPSERT_CACHE.clear()


async def upsert_embedding_profile(
    db: "Any",
    embedding_service: "Any",
    profile_id: str,
) -> None:
    """Upsert a row into ``embedding_profiles`` for the given profile id.

    Pulls human-readable fields from the service's ``describe()``
    method. Idempotent — second call for the same id is a no-op via
    the in-process cache, so the storage hot path stays cheap.

    Failure is always swallowed and logged at DEBUG. The registry
    table is operator-visibility only; storage and kNN don't read it.
    A failed upsert never blocks message persistence.
    """
    if not profile_id or profile_id in _PROFILE_UPSERT_CACHE:
        return
    if not hasattr(embedding_service, "describe"):
        return
    try:
        profile = embedding_service.describe()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Embedding service describe() raised: %s", exc)
        return
    if profile is None:
        return

    backend_type = getattr(db, "backend_type", None)
    try:
        if backend_type == "postgres":
            await db.execute(
                """INSERT INTO embedding_profiles
                       (id, provider, model, dim, space_id, normalized)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT (id) DO NOTHING""",
                (
                    profile.profile_id,
                    profile.provider,
                    profile.model,
                    profile.dim,
                    profile.space_id,
                    bool(profile.normalized),
                ),
            )
        else:
            # SQLite. Boolean → INTEGER (0/1).
            await db.execute(
                """INSERT OR IGNORE INTO embedding_profiles
                       (id, provider, model, dim, space_id, normalized)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    profile.profile_id,
                    profile.provider,
                    profile.model,
                    profile.dim,
                    profile.space_id,
                    1 if profile.normalized else 0,
                ),
            )
        _PROFILE_UPSERT_CACHE.add(profile.profile_id)
    except Exception as exc:
        logger.debug(
            "Embedding profile registry upsert failed (non-fatal) for %s: %s",
            profile_id, exc,
        )


class EmbeddingProfileRow(SovereignBase):
    """Registry row: one per (provider, model, dim, space, normalized) seen.

    The id is the 12-char deterministic hash from
    :class:`~kestrel_sovereign.llm.embedding_service.EmbeddingProfile.profile_id`,
    so this table is the canonical lookup from id → human-readable
    fields. Storage callers don't need to JOIN — they stamp the id on
    embedded rows directly. The registry is for operator visibility
    (``kestrel-sovereign embeddings audit``).
    """

    __tablename__ = "embedding_profiles"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    space_id: Mapped[str] = mapped_column(Text, nullable=False)
    normalized: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
