"""Constitution reanchor — write path for ``kestrel constitution reanchor``.

Drift detection ships in :mod:`kestrel_sovereign.doctor`. This module is
the writer side: when an agent's anchored ``constitution_hash`` no longer
matches the canonical file on disk, this helper updates **all five**
DB locations in a single AsyncStorage session, plus a timestamped
file-level backup so a botched run is recoverable.

The five places inception writes the constitution to (per
``inception_service.py:388``):

  1. ``files`` table — encrypted blob keyed by SHA256 hash.
  2. ``graph_nodes`` — a ``document`` node with id = hash.
  3. ``agent.properties.constitution_hash`` — the agent's pointer.
  4. ``graph_edges`` — a ``governed_by`` edge: agent_did → hash.
  5. ``document_chunks`` — RAG-indexed chunks keyed by file_hash.

We also append an audit entry at ``agent.properties.constitution_reanchor``
(format matches the runtime ``!reanchor-constitution`` chat command for
consistency: timestamp + old_hash + new_hash + source_path).

We do NOT delete the old document node or the old file blob — they're
retained for audit. Only the ``governed_by`` edge and the RAG chunks
move to the new hash.

Pre-flight: caller MUST ensure the agent isn't running. SQLite WAL
locking would otherwise corrupt mid-write.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kestrel_sovereign.storage import AsyncStorage, GraphNode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReanchorResult:
    """Outcome of :func:`reanchor_constitution`.

    Exactly one of ``unchanged`` / ``drift_unforced`` / ``reanchored`` /
    ``error`` is True. The CLI dispatches messaging on this.
    """

    agent_name: str
    db_path: Path
    canonical_path: Path
    old_hash: str | None
    new_hash: str | None
    backup_path: Path | None
    unchanged: bool = False
    drift_unforced: bool = False
    reanchored: bool = False
    error: str | None = None


async def reanchor_constitution(
    *,
    agent_name: str,
    agent_dir: Path,
    canonical_path: Path,
    force: bool,
    authorization: str = "kestrel constitution reanchor",
) -> ReanchorResult:
    """Reanchor one agent to the current canonical constitution.

    Read order matters: we open the DB twice. First (read-only) to
    discover whether drift exists; if no drift OR no ``--force``, we
    return without ever taking the destructive backup. Only when we're
    actually going to write do we copy the DB aside and reopen the
    storage layer for mutation.

    Args:
        agent_name: Display name (used for messages and backup naming).
        agent_dir: Agent's data directory (contains ``kestrel_prime.db``).
        canonical_path: On-disk constitution to anchor against.
        force: Required for any write. Without it, drift is reported
            but the DB is not touched.
        authorization: Free-form string stored in the audit record so
            future readers know who performed this reanchor.
    """
    db_path = agent_dir / "kestrel_prime.db"
    if not db_path.exists():
        return ReanchorResult(
            agent_name=agent_name,
            db_path=db_path,
            canonical_path=canonical_path,
            old_hash=None,
            new_hash=None,
            backup_path=None,
            error=f"Agent database not found at {db_path}",
        )

    try:
        new_content = canonical_path.read_bytes()
    except OSError as exc:
        return ReanchorResult(
            agent_name=agent_name,
            db_path=db_path,
            canonical_path=canonical_path,
            old_hash=None,
            new_hash=None,
            backup_path=None,
            error=f"Cannot read canonical constitution at {canonical_path}: {exc}",
        )

    new_hash = hashlib.sha256(new_content).hexdigest()

    old_hash, agent_did = await _read_agent_anchor(db_path)
    if old_hash is None:
        return ReanchorResult(
            agent_name=agent_name,
            db_path=db_path,
            canonical_path=canonical_path,
            old_hash=None,
            new_hash=new_hash,
            backup_path=None,
            error=(
                "Agent has no constitution_hash property. "
                "Re-incept the agent rather than reanchoring."
            ),
        )

    if old_hash == new_hash:
        return ReanchorResult(
            agent_name=agent_name,
            db_path=db_path,
            canonical_path=canonical_path,
            old_hash=old_hash,
            new_hash=new_hash,
            backup_path=None,
            unchanged=True,
        )

    if not force:
        return ReanchorResult(
            agent_name=agent_name,
            db_path=db_path,
            canonical_path=canonical_path,
            old_hash=old_hash,
            new_hash=new_hash,
            backup_path=None,
            drift_unforced=True,
        )

    backup_path = _backup_db(db_path)
    logger.info("Backed up agent DB to %s before reanchor", backup_path)

    try:
        await _write_reanchor(
            db_path=db_path,
            agent_did=agent_did,
            old_hash=old_hash,
            new_hash=new_hash,
            new_content=new_content,
            canonical_path=canonical_path,
            authorization=authorization,
        )
    except Exception as exc:  # noqa: BLE001 — surface the underlying error verbatim
        logger.exception("Reanchor failed; backup retained at %s", backup_path)
        return ReanchorResult(
            agent_name=agent_name,
            db_path=db_path,
            canonical_path=canonical_path,
            old_hash=old_hash,
            new_hash=new_hash,
            backup_path=backup_path,
            error=f"Reanchor failed mid-write ({exc!r}). DB backup at {backup_path}",
        )

    return ReanchorResult(
        agent_name=agent_name,
        db_path=db_path,
        canonical_path=canonical_path,
        old_hash=old_hash,
        new_hash=new_hash,
        backup_path=backup_path,
        reanchored=True,
    )


async def _read_agent_anchor(db_path: Path) -> tuple[str | None, str]:
    """Return ``(constitution_hash, agent_did)`` from the DB.

    Read-only — safe to call before deciding whether to touch the DB.
    Returns ``(None, "")`` if the agent node has no anchored hash.
    """
    async with AsyncStorage(str(db_path)) as storage:
        agent_nodes = await storage.graph.get_nodes_by_type("agent")
        if not agent_nodes:
            return None, ""
        agent = agent_nodes[0]
        return agent.properties.get("constitution_hash"), agent.node_id


async def _write_reanchor(
    *,
    db_path: Path,
    agent_did: str,
    old_hash: str,
    new_hash: str,
    new_content: bytes,
    canonical_path: Path,
    authorization: str,
) -> None:
    """Apply the five-location reanchor atomically.

    Wrapped in ``storage.db.transaction()``: every mutation below is
    a single SQLite transaction with automatic rollback on exception.
    Without this, the underlying backend auto-commits each call and
    a mid-write failure (RAG embedding, decode, edge deletion, the
    final node update) would leave the DB partially reanchored.
    The file-level DB backup the caller takes is the *outer* safety
    net; this transaction is the *inner* one and is what makes
    "reanchor is atomic" actually true.

    Order matters within the transaction:
      1. Store the new file blob first (idempotent INSERT OR IGNORE
         on content_hash). Doing this last would risk an "edge points
         at a hash that has no file" inconsistency under partial
         visibility.
      2. Add the new graph document node (idempotent upsert on node_id).
      3. Replace the governed_by edge: add new first, then delete old —
         so a concurrent reader inside the transaction (if any) never
         sees zero governing constitutions.
      4. Re-index RAG: chunk the new content, then drop the old chunks
         (same "always have something" reasoning).
      5. Update the agent node's properties last — that's the pointer
         everyone reads, so flipping it is the conceptual commit.

    If any step raises, the context manager rolls back and the DB is
    byte-identical to its pre-transaction state. The caller's file-
    level backup remains untouched and available either way.
    """
    async with AsyncStorage(str(db_path)) as storage:
        async with storage.db.transaction():
            # 1. File blob (encrypted at rest if KESTREL_DATA_KEY is set).
            stored_hash = await storage.files.store_file(
                new_content, "KESTREL_CONSTITUTION.md"
            )
            if stored_hash != new_hash:
                # store_file computes its own SHA256; if it disagrees with
                # ours something is profoundly wrong (different encoding,
                # corruption). Fail loudly — the transaction will roll back.
                raise RuntimeError(
                    f"File store hash mismatch: stored {stored_hash}, expected {new_hash}"
                )

            # 2. Document graph node for the new constitution.
            await storage.graph.add_node(
                GraphNode(
                    node_id=new_hash,
                    node_type="document",
                    label="KESTREL_CONSTITUTION",
                    properties={
                        "hash": new_hash,
                        "type": "Constitution",
                        "created_at": _now_iso(),
                    },
                )
            )

            # 3. Replace the governed_by edge — add new first.
            await storage.graph.add_edge(agent_did, new_hash, "governed_by")
            await storage.graph.delete_edge(agent_did, old_hash, "governed_by")

            # 4. Re-index RAG.
            await storage.rag.chunk_document(
                file_hash=new_hash,
                content=new_content.decode("utf-8"),
                chunk_size=500,
                compute_embeddings=True,
            )
            await storage.rag.delete_chunks_for_file(old_hash)

            # 5. Update the agent's pointer + audit record.
            agent_nodes = await storage.graph.get_nodes_by_type("agent")
            if not agent_nodes:
                raise RuntimeError("Agent node disappeared mid-reanchor")
            agent = agent_nodes[0]
            agent.properties["constitution_hash"] = new_hash
            agent.properties["constitution_reanchor"] = {
                "timestamp": _now_iso(),
                "old_hash": old_hash,
                "new_hash": new_hash,
                "source_path": str(canonical_path),
                "authorization": authorization,
            }
            await storage.graph.add_node(agent)


def _backup_db(db_path: Path) -> Path:
    """Copy the agent DB to a timestamped sibling. ``shutil.copy2``
    preserves mtime + permissions so the backup is restorable by
    ``cp`` / ``mv`` without surprises.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup = db_path.with_name(f"{db_path.name}.backup-{ts}")
    shutil.copy2(db_path, backup)
    return backup


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
