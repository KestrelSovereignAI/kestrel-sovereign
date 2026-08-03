"""Reconcile an agent's birth record with the database its runtime reads.

Inception writes the birth record — the agent node, the constitution node, the
``governed_by`` edge, the constitution files and their RAG chunks — into the
database it opened. When ``kestrel create`` is not handed a ``database=``, that
is a fresh SQLite at ``<agent_dir>/kestrel_prime.db``. A host configured with
``KESTREL_DB_BACKEND=postgres`` then boots the agent against PostgreSQL, where
none of it exists (#2871). The agent came up unnamed, with no
``bootstrap_state`` and nothing in Constitutional RAG to retrieve, while
``/health`` reported ``ok``.

Why the local SQLite is not simply removed: twelve places across seven modules
read ``(dir / "kestrel_prime.db").exists()`` as the fact that a directory *is*
an agent — ``agent_config.py`` alone resolves it in five branches. Deleting it
makes a PostgreSQL-created agent unrecognisable as an agent at all. So the file
stays as the local anchor and the birth record is *copied* from it into the
runtime database.

Why this runs at boot rather than inside inception: replication has to be
retryable. ``create_kestrel_identity_async`` interleaves durable identity
writes with embedding computation, so a copy performed inside it either widens
a transaction across model inference (holding SQLite's write lock, #2660) or
half-commits with no one left to finish the job — inception has already
created the anchor, so the next ``kestrel create`` refuses and nothing ever
repairs it. Running here makes every boot a retry, and makes the repair of an
agent already carrying a fabricated placeholder the same code path as the
prevention of a new one.

Nothing here computes an embedding. The vectors were computed once at
inception and are copied verbatim, so the copy is pure I/O and can safely be
one transaction on the target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kestrel_sovereign.storage import GraphNode
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)

#: The properties ``_ensure_agent_node_present`` gives a node it fabricates for
#: a genuinely new agent. An inception record always carries more than this
#: (``name``, ``constitution_hash``, ``bootstrap_state``, …), so an exact match
#: identifies a placeholder with no false positives on a real birth record.
_PLACEHOLDER_PROPERTIES = {"initialBalance": "100.0"}

GOVERNED_BY = "governed_by"


def is_fabricated_placeholder(node: Optional["GraphNode"], agent_did: str) -> bool:
    """Return whether ``node`` is a boot-fabricated stand-in, not a birth record.

    Matched on the exact shape ``_ensure_agent_node_present`` writes rather than
    on the absence of good properties: an old but genuine agent node might lack
    any single property, and refusing to boot such an agent would be a
    regression. A node that matches this shape was written by exactly one code
    path and never by inception.
    """
    if node is None:
        return False
    if node.label != f"Agent {agent_did}":
        return False
    return dict(node.properties or {}) == _PLACEHOLDER_PROPERTIES


@dataclass
class BirthRecordDivergence:
    """What the runtime database is missing relative to the local anchor."""

    reasons: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.reasons)

    def describe(self) -> str:
        return "; ".join(self.reasons)


@dataclass
class ReplicationResult:
    """What a replication pass actually wrote."""

    nodes: int = 0
    edges: int = 0
    files: int = 0
    chunks: int = 0
    #: Files the anchor claims this agent owns but whose bytes are not there —
    #: a dangling ``file_owners`` row. Recorded rather than dropped: a partial
    #: copy that reports success is the defect this module exists to remove.
    #: (A file present but undecryptable raises instead; a wrong
    #: ``KESTREL_DATA_KEY`` is not something to carry on past.)
    skipped_files: List[str] = field(default_factory=list)

    def describe(self) -> str:
        base = (
            f"{self.nodes} nodes, {self.edges} edges, {self.files} files, "
            f"{self.chunks} chunks"
        )
        if self.skipped_files:
            base += f" ({len(self.skipped_files)} file(s) unreadable and skipped)"
        return base


def local_anchor_path(storage_path: Optional[str]) -> Optional[Path]:
    """Return the agent's local inception database, if it is on disk."""
    if not storage_path:
        return None
    path = Path(storage_path)
    return path if path.exists() else None


def runtime_database_is_the_anchor(
    runtime_db: "AsyncDatabase", anchor: Path
) -> bool:
    """Return whether the runtime is already reading the anchor itself.

    The ordinary SQLite deployment: inception's database and the runtime's
    database are the same file, so there is nothing to reconcile and this
    module must be entirely inert. Decided on the file the runtime actually
    opened rather than on ``KESTREL_DB_BACKEND``, so a host pointed at some
    other SQLite file is still reconciled.
    """
    if runtime_db.backend_type != "sqlite":
        return False
    db_path = getattr(getattr(runtime_db, "backend", None), "db_path", None)
    if not db_path or db_path == ":memory:":
        return False
    try:
        return Path(db_path).resolve() == anchor.resolve()
    except OSError:  # pragma: no cover - unresolvable path
        return str(db_path) == str(anchor)


async def diagnose_birth_record(
    *,
    runtime_db: "AsyncDatabase",
    anchor_db: "AsyncDatabase",
    agent_did: str,
) -> BirthRecordDivergence:
    """Compare the runtime database against the anchor's birth record.

    Cheap enough to run on every boot: three lookups and two counts. Returns an
    empty divergence — the overwhelmingly common case — when the runtime
    database already holds the record.
    """
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore

    divergence = BirthRecordDivergence()

    anchor_graph = AsyncGraphStore(anchor_db)
    anchor_node = await anchor_graph.get_node(agent_did)
    if anchor_node is None:
        # No birth record in the anchor either. There is nothing to copy, and
        # saying so is not the same as saying the runtime record is fine — the
        # caller's refusal is what covers that.
        return divergence

    runtime_graph = AsyncGraphStore(runtime_db)
    runtime_node = await runtime_graph.get_node(agent_did)
    if runtime_node is None:
        divergence.reasons.append("agent node absent from the runtime database")
    elif is_fabricated_placeholder(runtime_node, agent_did):
        divergence.reasons.append(
            "runtime agent node is a boot-fabricated placeholder"
        )

    anchor_edges = {
        (edge.source_id, edge.target_id, edge.label)
        for edge in await anchor_graph.get_edges(agent_did, direction="out")
    }
    if anchor_edges:
        runtime_edges = {
            (edge.source_id, edge.target_id, edge.label)
            for edge in await runtime_graph.get_edges(agent_did, direction="out")
        }
        missing = anchor_edges - runtime_edges
        if missing:
            labels = sorted({label for _, _, label in missing})
            divergence.reasons.append(
                f"edges missing from the runtime database: {', '.join(labels)}"
            )

    anchor_chunks = await _owned_chunk_count(anchor_db, agent_did)
    if anchor_chunks:
        runtime_chunks = await _owned_chunk_count(runtime_db, agent_did)
        if runtime_chunks < anchor_chunks:
            divergence.reasons.append(
                f"constitution chunks: anchor has {anchor_chunks}, "
                f"runtime database has {runtime_chunks}"
            )

    return divergence


async def _owned_chunk_count(db: "AsyncDatabase", agent_did: str) -> int:
    row = await db.fetchone(
        "SELECT COUNT(*) FROM document_chunk_owners WHERE agent_id = ?",
        (agent_did,),
    )
    return int(row[0]) if row else 0


async def replicate_birth_record(
    *,
    runtime_db: "AsyncDatabase",
    anchor_db: "AsyncDatabase",
    agent_did: str,
    llm_service: Any = None,
) -> ReplicationResult:
    """Copy the agent's birth record from the anchor into the runtime database.

    Idempotent. Nodes and edges are whole-row upserts, file rows are content
    addressed, and chunks are replaced per file rather than appended — so an
    interrupted pass is repaired by the next one instead of accumulating
    duplicates.

    The copy is one transaction on the target: every write here is I/O against
    rows that already exist in the anchor, with no model inference inside the
    span, so the objection that kept this out of inception does not apply.
    """
    from kestrel_sovereign.storage.async_file_store import AsyncFileStore
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
    from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore

    # Bound to the agent on both sides, exactly as inception binds them
    # (inception_service.py: ``graph.bind_agent`` / ``files.bind_agent`` before
    # the first write). An unbound writer would record different ownership than
    # the record it is copying, so the copy would not be the same record.
    anchor_graph = AsyncGraphStore(anchor_db, agent_id=agent_did)
    anchor_files = AsyncFileStore(anchor_db, agent_id=agent_did)
    anchor_rag = AsyncRAGStore(anchor_db, agent_id=agent_did)

    runtime_graph = AsyncGraphStore(runtime_db, agent_id=agent_did)
    runtime_files = AsyncFileStore(runtime_db, agent_id=agent_did)
    runtime_rag = AsyncRAGStore(
        runtime_db, llm_service=llm_service, agent_id=agent_did
    )

    agent_node = await anchor_graph.get_node(agent_did)
    if agent_node is None:
        raise ValueError(
            f"The local anchor holds no birth record for {agent_did}; "
            "there is nothing to replicate."
        )

    edges = await anchor_graph.get_edges(agent_did, direction="out")
    # Read the whole record before opening the write transaction: the anchor is
    # a second connection to a second database, and a read on it inside the
    # target's transaction would hold that transaction open across it.
    #
    # A target this agent owns (its constitution node) is copied. A target owned
    # only by somebody else — a ``spawned_by`` parent living in another tenant —
    # is not; inception does not copy it either, and claiming it here would be
    # this agent writing over another's row. The edge still has to be recorded,
    # so those go through the trusted cross-agent writer.
    copyable: set = set()
    targets: List["GraphNode"] = []
    for edge in edges:
        if edge.target_id == agent_did or edge.target_id in copyable:
            continue
        owners = {
            row[0]
            for row in await anchor_db.fetchall(
                "SELECT agent_id FROM graph_node_owners WHERE node_id = ?",
                (edge.target_id,),
            )
        }
        if owners and agent_did not in owners:
            continue
        target = await anchor_graph.get_node(edge.target_id)
        if target is None:
            continue
        copyable.add(edge.target_id)
        targets.append(target)

    file_rows = await anchor_db.fetchall(
        "SELECT content_hash, original_name FROM file_owners WHERE agent_id = ?",
        (agent_did,),
    )
    payloads = []
    for content_hash, original_name in file_rows:
        content = await anchor_files.retrieve_file(content_hash)
        metadata = await anchor_files.get_file_metadata(content_hash) or {}
        chunks = await anchor_rag.read_indexed_chunks(content_hash)
        payloads.append((content_hash, original_name, content, metadata, chunks))

    result = ReplicationResult()

    async with runtime_db.transaction():
        # The agent node and everything it is governed by land together. A
        # present agent node makes every later boot treat inception as done, so
        # an agent recorded without its governing edge is never repaired
        # (#2867) — the same reason inception commits these as one unit.
        for node in targets:
            await runtime_graph.add_node(node)
            result.nodes += 1
        await runtime_graph.add_node(agent_node)
        result.nodes += 1
        for edge in edges:
            if edge.target_id == agent_did or edge.target_id in copyable:
                await runtime_graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    edge.label,
                    properties=edge.properties,
                )
            else:
                await runtime_graph.add_trusted_cross_agent_edge(
                    edge.source_id,
                    edge.target_id,
                    edge.label,
                    properties=edge.properties,
                )
            result.edges += 1

        for content_hash, original_name, content, metadata, chunks in payloads:
            if content is None:
                result.skipped_files.append(content_hash)
                logger.error(
                    "Birth-record file %s (%s) could not be read from the local "
                    "anchor; its chunks will not be replicated.",
                    content_hash, original_name,
                )
                continue
            # ``enc`` describes how the *source* row was stored. The target
            # re-encrypts under its own configuration, so carrying the flag
            # across would mark a plaintext row as encrypted.
            metadata.pop("enc", None)
            await runtime_files.store_file(
                content, original_name or content_hash, metadata=metadata,
            )
            result.files += 1

            if chunks:
                result.chunks += await runtime_rag.store_precomputed_chunks(
                    content_hash, chunks,
                )

    return result
