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

What is NOT copied: the ``conversation_history`` row inception writes when the
genesis audit passes. The audit's authoritative receipt is the ``genesis_audit``
property on the agent node, which ``agent/constitution.py`` reads and which does
cross; the conversation row is a second, narrative witness. Carrying it would
mean going through ``AsyncConversationStore.add_conversation``, which computes
an embedding — putting model inference back inside this transaction, the one
thing the design is built to avoid. Tracked on #2871 rather than done badly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
SPAWNED_BY = "spawned_by"


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

    def describe(self) -> str:
        return (
            f"{self.nodes} nodes, {self.edges} edges, {self.files} files, "
            f"{self.chunks} chunks"
        )


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


async def diagnose_runtime_birth_record(
    *,
    runtime_db: "AsyncDatabase",
    agent_did: str,
) -> BirthRecordDivergence:
    """Report what the runtime database is missing, WITHOUT consulting the anchor.

    This is the question that actually matters — "can this agent be who it is?"
    — and answering it from the runtime alone is what keeps a healthy
    PostgreSQL host independent of a leftover SQLite file it no longer needs.
    Opening the anchor runs ``_init_schema`` (creates, ALTERs, ownership
    backfills), so a corrupt or read-only ``kestrel_prime.db`` would otherwise
    refuse a boot whose runtime record is perfectly complete.

    Complete means: a real agent node (not a boot-fabricated placeholder), a
    ``governed_by`` edge, a readable node at the other end of it, and at least
    one retrievable constitution chunk. The target-node check is not
    redundant — a bound ``get_node`` returns None for a node with no ownership
    witness as well as for one that does not exist, so an edge can land while
    the node it names does not: "recorded but not governed" (#2867), which no
    later pass would notice.

    The store is BOUND to the agent, so this measures exactly what the running
    agent will later read through its own storage — and exactly what
    :func:`replicate_birth_record` is able to write. An unbound verifier sees
    rows the bound copier cannot produce and would refuse forever.
    """
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore

    divergence = BirthRecordDivergence()
    graph = AsyncGraphStore(runtime_db, agent_id=agent_did)

    node = await graph.get_node(agent_did)
    if node is None:
        divergence.reasons.append("agent node absent from the runtime database")
    elif is_fabricated_placeholder(node, agent_did):
        divergence.reasons.append(
            "runtime agent node is a boot-fabricated placeholder"
        )

    governing = [
        edge
        for edge in await graph.get_edges(agent_did, direction="out")
        if edge.label == GOVERNED_BY
    ]
    if not governing:
        divergence.reasons.append("no governed_by edge in the runtime database")
    else:
        for edge in governing:
            if await graph.get_node(edge.target_id) is None:
                divergence.reasons.append(
                    f"governed_by names {edge.target_id[:12]}… but no such node "
                    "is readable in the runtime database"
                )

    if await _owned_chunk_count(runtime_db, agent_did) == 0:
        divergence.reasons.append(
            "no retrievable constitution chunks in the runtime database"
        )

    return divergence


async def diagnose_birth_record(
    *,
    runtime_db: "AsyncDatabase",
    anchor_db: "AsyncDatabase",
    agent_did: str,
) -> BirthRecordDivergence:
    """Report what the runtime database is missing that the anchor can supply.

    Runs after :func:`diagnose_runtime_birth_record` has already said the
    runtime record is incomplete. It answers the narrower question the copier
    needs: which specific rows are absent.

    Deliberately one-directional. A runtime node that DIFFERS from the anchor is
    not a divergence — the anchor is frozen at inception while the runtime node
    goes on living (a completed genesis audit, a reanchored
    ``constitution_hash``, an avatar hash). Only absence is repairable, and
    treating difference as damage would undo signed constitutional amendments.
    """
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore

    divergence = BirthRecordDivergence()

    anchor_graph = AsyncGraphStore(anchor_db, agent_id=agent_did)
    anchor_node = await anchor_graph.get_node(agent_did)
    if anchor_node is None:
        # No birth record in the anchor either. There is nothing to copy, and
        # saying so is not the same as saying the runtime record is fine — the
        # caller's refusal is what covers that.
        return divergence

    runtime_graph = AsyncGraphStore(runtime_db, agent_id=agent_did)
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


async def _owned_chunk_count(
    db: "AsyncDatabase", agent_did: str, file_hash: Optional[str] = None,
) -> int:
    """Count the chunks replication can actually move — and RAG can retrieve.

    Joined through ``file_owners`` for the same reason the graph stores above
    are bound: a chunk-owner row whose file this agent does not own is outside
    ``AsyncRAGStore``'s tenant scope, so the copier will never write it and the
    agent could never retrieve it. Counting it would make the post-copy check
    demand a row nothing can produce.

    ``file_hash`` narrows it to one document. The post-copy assertion uses that
    form: a whole-tenant total cannot fall short once the agent owns unrelated
    chunks, so it would stop being able to detect anything.
    """
    sql = (
        "SELECT COUNT(*) FROM document_chunk_owners owners "
        "JOIN document_chunks chunks ON chunks.chunk_id = owners.chunk_id "
        "JOIN file_owners files "
        "  ON files.content_hash = chunks.file_hash "
        " AND files.agent_id = owners.agent_id "
        "WHERE owners.agent_id = ?"
    )
    params: tuple = (agent_did,)
    if file_hash is not None:
        sql += " AND chunks.file_hash = ?"
        params += (file_hash,)
    row = await db.fetchone(sql, params)
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
    # A bound store only sees edges carrying this agent's ownership witness, so
    # an anchor edge without one is invisible to the copier — it would commit an
    # agent node whose ``constitution_hash`` names a node it never wrote, i.e.
    # the "recorded but not governed" state #2867 exists to prevent, and no
    # later boot would notice. Refuse with the reason instead. (The #2649
    # ownership backfill skips edges it cannot prove a common owner for, so this
    # is a shape a real anchor can carry.)
    all_edges = {
        (edge.source_id, edge.target_id, edge.label)
        for edge in await AsyncGraphStore(anchor_db).get_edges(
            agent_did, direction="out",
        )
    }
    unwitnessed = {
        (source, target, label)
        for source, target, label in all_edges - {
            (e.source_id, e.target_id, e.label) for e in edges
        }
        # Scoped to the birth record's own edges. An unwitnessed edge of some
        # other kind is a gap in the #2649 backfill, not a reason to refuse an
        # identity — this guard exists to stop a SILENTLY ungoverned agent, not
        # to audit the anchor's whole graph.
        if label in (GOVERNED_BY, SPAWNED_BY)
    }
    if unwitnessed:
        raise ValueError(
            f"the local anchor holds {sorted(l for _, _, l in unwitnessed)} "
            f"edges for {agent_did} with no ownership witness for it; "
            "replicating would record the agent without them. Repair the "
            "anchor's graph_edge_owners rows before booting against another "
            "database."
        )
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
            # Deliberately another tenant's node — a ``spawned_by`` parent.
            # Inception does not copy it either; the edge alone is the record.
            continue
        target = await anchor_graph.get_node(edge.target_id)
        if target is None:
            # Not readable through the bound store: either absent, or present
            # with no ownership witness. For the constitution that is the birth
            # record's own node, and copying the edge without it would record an
            # agent governed by nothing (#2867) — refuse, same as an
            # unwitnessed edge. A ``spawned_by`` parent is legitimately absent.
            if edge.label == GOVERNED_BY:
                raise ValueError(
                    f"the local anchor's {GOVERNED_BY} edge for {agent_did} "
                    f"names {edge.target_id[:12]}… but that node is not "
                    "readable there (absent, or carrying no ownership witness); "
                    "the birth record cannot be replicated from it."
                )
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
        if content is None:
            # A file_owners row with no file bytes. The anchor cannot produce
            # this part of the record and no later boot can either, so say so
            # here — before a write transaction is opened — rather than
            # committing a record whose chunks can never arrive.
            raise ValueError(
                f"the local anchor claims {agent_did} owns file {content_hash} "
                f"({original_name}) but holds no bytes for it; the birth record "
                "cannot be replicated from it."
            )
        metadata = await anchor_files.get_file_metadata(content_hash) or {}
        chunks = await anchor_rag.read_indexed_chunks(content_hash)
        payloads.append((content_hash, original_name, content, metadata, chunks))

    result = ReplicationResult()

    async with runtime_db.transaction():
        # Repair what is ABSENT; never overwrite what is present. The anchor is
        # frozen at inception and the runtime node goes on living — a completed
        # genesis audit, a reanchored constitution_hash, an avatar hash — so a
        # whole-row copy would revert durable post-inception state. A boot whose
        # only complaint was a chunk count would undo a signed constitutional
        # amendment and drop the agent back into GENESIS AUDIT PENDING.
        #
        # Files BEFORE nodes, which is inception's order (store_file at
        # inception_service.py:873, add_node at :977) and is load-bearing on a
        # runtime database shared by more than one agent — the deployment
        # PostgreSQL exists for. The constitution node is content-addressed, so
        # every agent under the same constitution shares one row, and
        # ``add_node`` admits a second tenant's ownership witness on a shared
        # content node only if that tenant already owns the underlying file
        # (``owns_content_reference``). Writing nodes first makes the second
        # agent raise "Cannot overwrite a graph node owned by another agent",
        # roll the whole copy back, and never boot.
        for content_hash, original_name, content, metadata, _chunks in payloads:
            if await runtime_files.file_exists(content_hash):
                continue
            # ``enc`` describes how the *source* row was stored. The target
            # re-encrypts under its own configuration, so carrying the flag
            # across would mark a plaintext row as encrypted.
            metadata.pop("enc", None)
            await runtime_files.store_file(
                content, original_name or content_hash, metadata=metadata,
            )
            result.files += 1

        for node in targets:
            if await runtime_graph.get_node(node.node_id) is None:
                await runtime_graph.add_node(node)
                result.nodes += 1

        runtime_node = await runtime_graph.get_node(agent_did)
        if runtime_node is None or is_fabricated_placeholder(runtime_node, agent_did):
            await runtime_graph.add_node(agent_node)
            result.nodes += 1

        # The governing edge is written whenever it is missing, even when the
        # node was already present: "agent recorded but not governed" is the
        # #2867 state, and it is only ever repaired here.
        present_edges = {
            (edge.source_id, edge.target_id, edge.label)
            for edge in await runtime_graph.get_edges(agent_did, direction="out")
        }
        for edge in edges:
            if (edge.source_id, edge.target_id, edge.label) in present_edges:
                continue
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

        # Chunks last: ``store_precomputed_chunks`` refuses a file outside the
        # bound agent, so the file_owners rows written above are its precondition.
        expected: Dict[str, int] = {}
        for content_hash, _original_name, _content, _metadata, chunks in payloads:
            if not chunks:
                continue
            if await _owned_chunk_count(runtime_db, agent_did, content_hash):
                # Already indexed here. Re-writing would delete and re-insert
                # rows the agent may since have re-embedded under a newer model.
                continue
            expected[content_hash] = len(chunks)
            result.chunks += await runtime_rag.store_precomputed_chunks(
                content_hash, chunks,
            )

        # Verify BEFORE committing, against what was read from the anchor
        # rather than by re-reading it, so nothing on the anchor is touched
        # inside this transaction. A store can report a write it did not
        # durably make — that is the whole defect class this issue belongs to —
        # and half a birth record must not survive the check. Raising here
        # rolls the copy back, so the next boot retries from a clean state
        # instead of inheriting an agent node with no governing edge (#2867).
        written = await runtime_graph.get_node(agent_did)
        if written is None or is_fabricated_placeholder(written, agent_did):
            raise ValueError(
                f"the agent node for {agent_did} is not readable in the runtime "
                "database after writing it"
            )
        written_edges = {
            (edge.source_id, edge.target_id, edge.label)
            for edge in await runtime_graph.get_edges(agent_did, direction="out")
        }
        missing = {
            (edge.source_id, edge.target_id, edge.label) for edge in edges
        } - written_edges
        if missing:
            raise ValueError(
                f"edges missing after writing them: "
                f"{sorted(label for _, _, label in missing)}"
            )
        # The node an edge NAMES, not just the edge. A bound ``get_node``
        # returns None for a node with no ownership witness as well as for one
        # that does not exist, so an edge can land while its target does not —
        # committing "recorded but not governed" and then reporting healthy.
        for edge in edges:
            if edge.target_id == agent_did:
                continue
            if edge.label != GOVERNED_BY and edge.target_id not in copyable:
                continue  # a spawned_by parent lives in another tenant by design
            if await runtime_graph.get_node(edge.target_id) is None:
                raise ValueError(
                    f"{edge.label} names {edge.target_id[:12]}… but that node is "
                    "not readable in the runtime database after writing it"
                )
        for content_hash, count in expected.items():
            landed = await _owned_chunk_count(runtime_db, agent_did, content_hash)
            if landed < count:
                raise ValueError(
                    f"{count} chunks were written for {content_hash[:12]}… but "
                    f"only {landed} are readable in the runtime database"
                )

    return result
