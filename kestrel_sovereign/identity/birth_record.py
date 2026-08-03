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
    """What the runtime database is missing, split by what it costs.

    The distinction is load-bearing, and getting it wrong is how a fix for a
    degraded agent becomes a brick.

    ``identity`` — the runtime cannot say who this agent is: no agent node, or
    a boot-fabricated placeholder standing in for one. This is #2878's
    condition exactly, and boot refuses on it whether or not a local anchor
    exists to repair it from. It is also the only condition this module widens
    nothing for: an agent that boots today keeps booting.

    ``capability`` — the record exists but something it should carry does not:
    no governing edge, a governing edge naming an unreadable node, no
    retrievable constitution chunks. Replication repairs these whenever the
    anchor can supply them. When it cannot — a pre-#2649 anchor whose chunk
    ownership was never provable, a constitution the anchor never held —
    refusing would convert an agent that boots today into one that can never
    boot again, with no operator verb to fix it. So these are reported loudly
    (log + ``/health/detailed``) and the agent boots degraded.

    That is not a retreat to the original bug. #2871's defect was that the loss
    was SILENT while ``/health`` said ok. Naming it is the fix; refusing on it
    is a policy change with no recovery path, and belongs to whoever decides
    an ungoverned agent must not run at all.
    """

    identity: List[str] = field(default_factory=list)
    capability: List[str] = field(default_factory=list)

    @property
    def reasons(self) -> List[str]:
        return [*self.identity, *self.capability]

    def __bool__(self) -> bool:
        return bool(self.identity or self.capability)

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


async def _carry_embedding_profiles(
    *, anchor_db: "AsyncDatabase", runtime_db: "AsyncDatabase", payloads,
) -> None:
    """Copy the ``embedding_profiles`` rows the carried vectors refer to.

    A chunk's ``embedding_profile_id`` is a foreign key in spirit: kNN filters
    by the active profile, and the operator audit resolves it to a provider,
    model and dimension. Carrying vectors without their registry row leaves
    them searchable but unattributable.

    Best-effort by design: the registry is #1477 and later, so a database that
    predates it has no such table, and losing operator metadata must never fail
    an identity repair. Never write a row that is already there — a co-owner's
    profile description is theirs.
    """
    profile_ids = {
        chunk.profile_id
        for *_head, chunks in payloads
        for chunk in chunks
        if chunk.profile_id
    }
    if not profile_ids:
        return
    columns = "id, provider, model, dim, space_id, normalized"
    for profile_id in sorted(profile_ids):
        try:
            row = await anchor_db.fetchone(
                f"SELECT {columns} FROM embedding_profiles WHERE id = ?",
                (profile_id,),
            )
            if row is None:
                continue
            existing = await runtime_db.fetchone(
                "SELECT 1 FROM embedding_profiles WHERE id = ?", (profile_id,),
            )
            if existing:
                continue
            insert = (
                f"INSERT INTO embedding_profiles ({columns}) "
                "VALUES (?, ?, ?, ?, ?, ?)"
            )
            insert += (
                " ON CONFLICT (id) DO NOTHING"
                if runtime_db.backend_type == "postgres"
                else ""
            )
            if runtime_db.backend_type != "postgres":
                insert = insert.replace("INSERT INTO", "INSERT OR IGNORE INTO", 1)
            await runtime_db.execute(insert, tuple(row))
        except Exception as exc:  # pragma: no cover - registry is optional
            logger.info(
                "Could not carry embedding profile %s into the runtime "
                "database (%s); copied chunks keep their vectors but will "
                "report an unknown profile in the embeddings audit.",
                profile_id, exc,
            )
            return


def _insert_file_owner_sql(db: "AsyncDatabase") -> str:
    """Insert-if-absent for a ``file_owners`` witness, both backends."""
    if db.backend_type == "postgres":
        return (
            "INSERT INTO file_owners (content_hash, agent_id, original_name, "
            "metadata) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING"
        )
    return (
        "INSERT OR IGNORE INTO file_owners (content_hash, agent_id, "
        "original_name, metadata) VALUES (?, ?, ?, ?)"
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
        divergence.identity.append("agent node absent from the runtime database")
    elif is_fabricated_placeholder(node, agent_did):
        divergence.identity.append(
            "runtime agent node is a boot-fabricated placeholder"
        )

    governing = [
        edge
        for edge in await graph.get_edges(agent_did, direction="out")
        if edge.label == GOVERNED_BY
    ]
    if not governing:
        divergence.capability.append(
            "no governed_by edge in the runtime database"
        )
    else:
        for edge in governing:
            if await graph.get_node(edge.target_id) is None:
                divergence.capability.append(
                    f"governed_by names {edge.target_id[:12]}… but no such node "
                    "is readable in the runtime database"
                )

    # Scoped to the governing document, never the tenant total. Any other
    # indexed file — a note, an upload — would otherwise satisfy this and
    # report an agent with ZERO constitution chunks as complete: #2871's
    # original symptom, signed off by the check added to catch it.
    for governing_hash in governing_document_hashes(node, governing):
        if await _owned_chunk_count(runtime_db, agent_did, governing_hash) == 0:
            divergence.capability.append(
                f"no retrievable chunks for the governing constitution "
                f"{governing_hash[:12]}… in the runtime database"
            )

    return divergence


def governing_document_hashes(
    node: Optional["GraphNode"], governing_edges: List[Any],
) -> List[str]:
    """Return the document(s) whose chunks are this agent's constitution.

    The agent node's ``constitution_hash`` is authoritative when present — it
    is what ``agent/constitution.py`` verifies against and what a reanchor
    updates. The ``governed_by`` targets are the fallback for a node predating
    that property. Returns empty only when neither exists, in which case there
    is no constitution to demand chunks for.
    """
    anchored = (node.properties or {}).get("constitution_hash") if node else None
    if isinstance(anchored, str) and anchored:
        return [anchored]
    return [edge.target_id for edge in governing_edges]


async def anchor_holds_birth_record(
    *, anchor_db: "AsyncDatabase", agent_did: str,
) -> bool:
    """Return whether the anchor has an agent node this agent can copy.

    Separate from :func:`diagnose_birth_record` because that function returns
    an empty divergence for two opposite situations — "the anchor has nothing"
    and "everything already matches" — and the caller must tell them apart.
    """
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore

    graph = AsyncGraphStore(anchor_db, agent_id=agent_did)
    return await graph.get_node(agent_did) is not None


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
        divergence.identity.append("agent node absent from the runtime database")
    elif is_fabricated_placeholder(runtime_node, agent_did):
        divergence.identity.append(
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
            divergence.capability.append(
                f"edges missing from the runtime database: {', '.join(labels)}"
            )

    # Per governing document, not the tenant total: a whole-tenant comparison
    # cannot fall short once the agent owns unrelated chunks, so it would say
    # "no rows missing" in the very log line that reports a repair in progress.
    for governing_hash in governing_document_hashes(
        runtime_node or anchor_node,
        [e for e in await anchor_graph.get_edges(agent_did, direction="out")
         if e.label == GOVERNED_BY],
    ):
        anchor_chunks = await _owned_chunk_count(
            anchor_db, agent_did, governing_hash,
        )
        if not anchor_chunks:
            continue
        runtime_chunks = await _owned_chunk_count(
            runtime_db, agent_did, governing_hash,
        )
        if runtime_chunks < anchor_chunks:
            divergence.capability.append(
                f"chunks for {governing_hash[:12]}…: anchor has {anchor_chunks}, "
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
    # Deliberately no ``llm_service``: nothing here computes an embedding, and
    # threading one in would invite a caller to assume otherwise.
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
        runtime_db, agent_id=agent_did
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
    # Scoped to the birth record's own edges AND to the ones that actually
    # matter. An unwitnessed edge of another kind is a gap in the #2649
    # backfill; a stale unwitnessed ``governed_by`` alongside a witnessed
    # current one is what a pre-atomic reanchor leaves behind, and `doctor`
    # treats it as a warning. This guard exists to stop a SILENTLY ungoverned
    # agent, not to audit the anchor's whole graph — so it fires only when the
    # relation the agent actually needs has no witness at all.
    witnessed = {(e.source_id, e.target_id, e.label) for e in edges}
    anchored_hash = (agent_node.properties or {}).get("constitution_hash")
    unwitnessed_labels = set()
    for _source, target, label in all_edges - witnessed:
        if label == GOVERNED_BY:
            witnessed_governing = {t for _s, t, l in witnessed if l == GOVERNED_BY}
            if anchored_hash in witnessed_governing or (
                not anchored_hash and witnessed_governing
            ):
                continue  # the current governing edge IS witnessed; this is stale
            unwitnessed_labels.add(label)
        elif label == SPAWNED_BY:
            if any(l == SPAWNED_BY for _s, _t, l in witnessed):
                continue
            unwitnessed_labels.add(label)
    if unwitnessed_labels:
        raise ValueError(
            f"the local anchor holds {sorted(unwitnessed_labels)} edges for "
            f"{agent_did} with no ownership witness for it, and no witnessed "
            "one to use instead; replicating would record the agent without "
            "them. Repair the anchor's graph_edge_owners rows before booting "
            "against another database."
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

    # Ordered so every agent under one constitution takes the shared ``files``
    # and ``graph_nodes`` row locks in the same sequence. Unordered, the lock
    # order is whatever the anchor's scan returns, which is how concurrent
    # replications into one PostgreSQL would deadlock against each other.
    file_rows = await anchor_db.fetchall(
        "SELECT content_hash, original_name FROM file_owners "
        "WHERE agent_id = ? ORDER BY content_hash",
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
            unowned_row = await runtime_db.fetchone(
                "SELECT 1 FROM files f WHERE f.content_hash = ? AND NOT EXISTS ("
                "  SELECT 1 FROM file_owners o WHERE o.content_hash = f.content_hash"
                ")",
                (content_hash,),
            )
            if unowned_row:
                # The bytes are there with no owner at all, so the owner-scoped
                # ``file_exists`` cannot see them and ``store_file`` would raise
                # "Cannot claim an unowned legacy file" on every boot, forever
                # — the same permanent brick the node adoption below avoids.
                # Files are content addressed, so the anchor's bytes hashing to
                # this same content_hash IS the proof of what the row holds.
                await runtime_db.execute(
                    _insert_file_owner_sql(runtime_db),
                    (content_hash, agent_did, original_name or content_hash, None),
                )
                result.files += 1
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
            if await runtime_graph.get_node(node.node_id) is not None:
                continue
            existing_row = await runtime_db.fetchone(
                "SELECT 1 FROM graph_nodes WHERE node_id = ?", (node.node_id,),
            )
            if existing_row:
                # The row is there but this agent has no ownership witness for
                # it, so the bound read cannot see it. ``add_node`` would raise
                # "Cannot claim or overwrite an unowned graph node" on every
                # boot, forever. Taking the witness is the repair — but only on
                # the same proof ``add_node`` itself requires to admit a
                # co-owner of a shared content node: this agent owns the file
                # the node addresses. Asserted rather than assumed, because
                # "the file rows above already prove it" is false whenever the
                # anchor listed no files.
                owns_content = await runtime_db.fetchone(
                    "SELECT 1 FROM file_owners WHERE content_hash = ? AND agent_id = ?",
                    (node.node_id, agent_did),
                )
                if not owns_content:
                    raise ValueError(
                        f"{node.node_id[:12]}… exists in the runtime database "
                        f"with no ownership witness for {agent_did}, and this "
                        "agent does not own the file it addresses; refusing to "
                        "claim another tenant's node."
                    )
                from kestrel_sovereign.storage.async_graph_store import (
                    record_graph_node_owner,
                )
                await record_graph_node_owner(runtime_db, node.node_id, agent_did)
                result.nodes += 1
                continue
            await runtime_graph.add_node(node)
            result.nodes += 1

        runtime_node = await runtime_graph.get_node(agent_did)
        if runtime_node is None or is_fabricated_placeholder(runtime_node, agent_did):
            await runtime_graph.add_node(agent_node)
            result.nodes += 1

        # The governing edge is written whenever it is missing, even when the
        # node was already present: "agent recorded but not governed" is the
        # #2867 state, and it is only ever repaired here.
        runtime_edges = await runtime_graph.get_edges(agent_did, direction="out")
        present_edges = {
            (edge.source_id, edge.target_id, edge.label) for edge in runtime_edges
        }
        # What the RUNTIME currently says governs this agent. A reanchor
        # updates the runtime node and prunes the old edge; the anchor keeps
        # the original. Re-adding the anchor's stale target would leave two
        # governing constitutions — which `doctor` reports and only a signed
        # `constitution reanchor --force` can clear.
        runtime_node_now = await runtime_graph.get_node(agent_did)
        runtime_anchored_hash = (
            (runtime_node_now.properties or {}).get("constitution_hash")
            if runtime_node_now
            else None
        )
        # The edges this pass means the runtime to end up holding — what the
        # verification below is entitled to demand. An edge deliberately not
        # replicated must not then be reported missing.
        intended: set = set()
        for edge in edges:
            triple = (edge.source_id, edge.target_id, edge.label)
            if triple in present_edges:
                intended.add(triple)
                continue
            # Regardless of whether a governing edge is currently present. The
            # case that matters most is the one where it is ABSENT — a reanchor
            # that pruned the old edge and died before writing the new one — and
            # requiring a present edge would let exactly that case re-attach the
            # superseded constitution.
            if (
                edge.label == GOVERNED_BY
                and runtime_anchored_hash
                and edge.target_id != runtime_anchored_hash
            ):
                logger.info(
                    "Not replicating the anchor's governed_by edge to %s… for "
                    "%s: the runtime database is already governed by %s…, "
                    "which is what its agent node anchors.",
                    edge.target_id[:12], agent_did, runtime_anchored_hash[:12],
                )
                continue
            intended.add(triple)
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

        # The registry row for every profile the copied vectors were stamped
        # with. Without it ``kestrel embeddings audit`` reports them as
        # "unknown — not in embedding_profiles registry": the vectors would be
        # searchable but unattributable to the model that made them.
        await _carry_embedding_profiles(
            anchor_db=anchor_db, runtime_db=runtime_db, payloads=payloads,
        )

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
        missing = intended - written_edges
        if missing:
            raise ValueError(
                f"edges missing after writing them: "
                f"{sorted(label for _, _, label in missing)}"
            )
        # The node an edge NAMES, not just the edge. A bound ``get_node``
        # returns None for a node with no ownership witness as well as for one
        # that does not exist, so an edge can land while its target does not —
        # committing "recorded but not governed" and then reporting healthy.
        for _source, target, label in intended:
            if target == agent_did:
                continue
            if label != GOVERNED_BY and target not in copyable:
                continue  # a spawned_by parent lives in another tenant by design
            if await runtime_graph.get_node(target) is None:
                raise ValueError(
                    f"{label} names {target[:12]}… but that node is "
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
