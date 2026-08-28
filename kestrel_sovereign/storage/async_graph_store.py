"""
Async Graph Store for Kestrel Storage.

Provides async knowledge graph storage with nodes and edges.

Sortable-timestamp invariant
----------------------------
The ``query_nodes_by_type_and_property`` method supports ``created_since``
range filtering and ``ORDER BY created_at DESC`` sorting.  Both rely on
lexicographic SQL comparison, which produces correct results **only** when
every ``created_at`` value stored in ``graph_nodes.properties`` is a
UTC ISO-8601 string whose text sort order matches chronological order —
i.e. ``YYYY-MM-DDTHH:MM:SS+00:00``.

All code paths that persist ``created_at`` MUST use
``datetime.now(timezone.utc).isoformat()`` (or an equivalent that
produces a fixed-offset ``+00:00`` suffix, never a bare naive string).
"""
import hashlib
import json
import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional
from dataclasses import dataclass

from .async_database import AsyncDatabase
from .async_conversation_store import _rows_affected

logger = logging.getLogger(__name__)

_DELETE_ID_BATCH = 500
_MAX_ABSENT_NODE_ADVISORY_LOCKS = 128

#: A SHA-256 digest as this codebase writes them: lowercase hex, 64 chars.
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _has_only(properties: Dict[str, Any], allowed: frozenset) -> bool:
    """Whether ``properties`` carries no key outside ``allowed``.

    Spelled as a subset test on purpose. The obvious-looking
    ``set(properties) > allowed`` asks whether the keys are a *proper superset*
    of the allowed set, which is a different — and much weaker — question: swap
    an optional key for an unlisted one (drop ``created_at``, add
    ``source_path``) and the key set is no longer a superset at all, so the
    guard never fires and the unlisted key rides onto a row other tenants
    co-own. That is the exact leak these predicates exist to prevent.
    """
    return set(properties) <= allowed


#: The bounded public metadata a shared constitution anchor may carry.
_SHAREABLE_CONSTITUTION_KEYS = frozenset({"hash", "type", "created_at"})

#: Exactly the per-agent fields the release *before* #2893 wrote onto the
#: artifact node, and nothing else. Normalisation is licensed by knowing what
#: these are: they are that release's noise, so dropping them loses nothing.
#: A key outside this set has unknown provenance — a field a later release
#: added, or one an operator put there — and trimming it would be one tenant
#: silently deleting another's data on the way to co-owning the row.
_LEGACY_ARTIFACT_KEYS = frozenset({"source_path", "anchored_at", "verification"})

#: The bounded, content-derived metadata a shared reanchor artifact may carry.
_SHAREABLE_ARTIFACT_KEYS = frozenset(
    {
        "hash",
        "type",
        "artifact_type",
        "constitution_hash",
        "signer",
        "created_at",
    }
)


def _is_tz_aware_iso(value: Any) -> bool:
    """A timezone-qualified ISO timestamp, bounded in length."""
    if not isinstance(value, str) or len(value) > 64:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _is_shareable_constitution_properties(
    properties: Dict[str, Any], node_id: str
) -> bool:
    """Validate the bounded public metadata allowed on a shared anchor."""
    if not _has_only(properties, _SHAREABLE_CONSTITUTION_KEYS):
        return False
    if (
        properties.get("hash") != node_id
        or properties.get("type") != "Constitution"
    ):
        return False
    created_at = properties.get("created_at")
    if created_at is None:
        return True
    return _is_tz_aware_iso(created_at)


def _accepted_amendment_artifact_types() -> frozenset:
    """The ``artifact_type`` values a signed reanchor artifact may declare.

    A closed set, because this field is admitted onto a row several tenants
    share — and taken from the verifier's own constant rather than restated
    here, so the two cannot drift. Imported lazily to keep the storage layer
    import-light and free of a dependency on the constitution package.
    """
    from kestrel_sovereign.constitution.amendment_artifact import ARTIFACT_TYPE

    return frozenset({ARTIFACT_TYPE})


def _is_shareable_amendment_artifact_properties(
    properties: Dict[str, Any], node_id: str
) -> bool:
    """Validate the bounded, content-derived metadata on a shared artifact node.

    A Sovereign-signed reanchor artifact is content-addressed, so one file is
    one node id for a whole PostgreSQL fleet — and every field admitted here is
    fixed by the bytes that hash to that id: ``artifact_type``, ``signer``,
    ``constitution_sha256`` and ``created_at`` are all *signed* fields of the
    artifact (see ``canonical_amendment_bytes``). Two agents anchoring the same
    file therefore compute the same properties, which is what makes the node
    genuinely shareable rather than merely coincident.

    What is deliberately **not** here is everything per-agent: ``source_path``
    (an operator filesystem path), ``anchored_at`` (when *this* agent anchored
    it) and ``verification`` (the result of checking the signature against
    *this* agent's resolved trust root). Those live on the agent's own
    ``constitution_reanchor`` audit property, which already records all three
    (#2893). Putting them on a fleet-wide node was the defect: it made one
    tenant's paths reachable from another's row, which is why
    ``privacy_wrapper`` refused to treat this node type as content-free.
    """
    if not _has_only(properties, _SHAREABLE_ARTIFACT_KEYS):
        return False
    if (
        properties.get("hash") != node_id
        or properties.get("type") != "SignedConstitutionAmendment"
    ):
        return False
    if properties.get("artifact_type") not in _accepted_amendment_artifact_types():
        return False
    constitution_hash = properties.get("constitution_hash")
    if not isinstance(constitution_hash, str) or not _HEX64.fullmatch(
        constitution_hash
    ):
        return False
    signer = properties.get("signer")
    if not isinstance(signer, str) or len(signer) > 256:
        return False
    if not signer.startswith("did:"):
        return False
    created_at = properties.get("created_at")
    if created_at is None:
        return True
    # Bounded, but *not* required to be timezone-qualified — unlike the
    # constitution predicate above, which validates a field this codebase
    # writes. ``created_at`` here is a **signed** field of the artifact:
    # ``canonical_amendment_bytes`` covers it and ``verify_reanchor_artifact``
    # does not constrain its shape. Requiring more of it than the verifier does
    # would let an artifact govern the first agent and not the second — the
    # first commits, because a brand-new row is never checked against this
    # predicate, and the second rolls back. That is the fleet split this issue
    # exists to remove. The length cap is what keeps a shared row bounded;
    # anyone who can choose this value already holds the Sovereign key and
    # could choose ``signer`` too.
    return isinstance(created_at, str) and len(created_at) <= 64


def _canonical_shared_properties(
    properties: Dict[str, Any], allowed: frozenset
) -> Dict[str, Any]:
    """``properties`` with everything outside ``allowed`` dropped."""
    return {key: value for key, value in properties.items() if key in allowed}


def _is_normalisable_legacy_artifact(
    existing: Dict[str, Any], incoming: Dict[str, Any], node_id: str
) -> bool:
    """Whether a pre-#2893 artifact row can be normalised into a shared one.

    The release before this one wrote ``source_path``, ``anchored_at`` and
    ``verification`` onto the artifact node. A fleet that has already reanchored
    its first agent therefore *has* such a row — and it is the exact state this
    issue exists to repair, so refusing to share it would fix the bug only for
    installations that never hit it. The second agent's reanchor rolled back
    with "Cannot overwrite a graph node owned by another agent".

    Only the fields that release actually wrote may be dropped
    (``_LEGACY_ARTIFACT_KEYS``). A row carrying anything else is not the legacy
    shape — it is a row someone or something else has written to — and joining
    it must not begin by deleting what is there.

    Normalising is safe precisely because the surviving fields are derived from
    the artifact bytes: if the legacy row's content-derived subset is byte-equal
    to what this writer computes from the same file, the extras are the previous
    release's per-agent noise and dropping them is what #2893 says should happen
    to them. It is a privacy improvement, not a rewrite of anybody's content —
    the fields that identify the *content* do not move.

    Everything else still applies: the caller has already established that this
    writer independently owns the underlying blob.
    """
    trimmed = _canonical_shared_properties(existing, _SHAREABLE_ARTIFACT_KEYS)
    if trimmed == existing:
        return False  # nothing to normalise; the ordinary path handles it
    if not set(existing) - _SHAREABLE_ARTIFACT_KEYS <= _LEGACY_ARTIFACT_KEYS:
        return False
    if not _is_shareable_amendment_artifact_properties(trimmed, node_id):
        return False
    return trimmed == _canonical_shared_properties(
        incoming, _SHAREABLE_ARTIFACT_KEYS
    )


@dataclass(frozen=True)
class _SharedContentShape:
    """One ``(node_type, label)`` a PostgreSQL fleet may hold a single row for.

    Two questions have to be asked of a shared row, and they are not the same
    question:

    * ``is_shareable`` — *could* a sibling tenant have computed these
      properties? That is a check on one property set in isolation.
    * ``identity_keys`` — do the stored row and the incoming node actually
      *agree*? Two property sets can each be shareable while describing
      different content. Admitting the second owner then leaves the first row
      standing and reports success to an agent that believes it stored the
      second — a record claiming what nobody observed.

    The two sets differ per shape, which is why this is a table rather than one
    rule: every field of a signed artifact is covered by the signature and must
    match byte for byte, but a constitution anchor's ``created_at`` is when
    *that* tenant first stored the document and legitimately differs across the
    fleet.

    What this layer does **not** promise, so nobody reads more into it: it
    enforces that a shared row's identity never *changes*, not that it ever
    matched the bytes. It cannot — deciding whether ``signer`` is really the
    artifact's signer means verifying a signature, which is
    ``verify_reanchor_artifact``'s job and the reanchor writers' to call. So a
    first write of wrong-but-well-formed properties still poisons that node id
    for the fleet, and a sole owner can reach the same state by deleting its
    row and recreating it. Both reduce to "the first writer lied", which is
    bounded upstream by the writers deriving properties from a *verified*
    artifact — not here.
    """

    #: Whether a property set is content-derived, and so co-ownable.
    is_shareable: Callable[[Dict[str, Any], str], bool]
    #: The subset of properties every co-owner must agree on.
    identity_keys: frozenset
    #: Whether a row written by the release *before* #2893 — carrying that
    #: release's per-agent fields — is normalised into the shared shape rather
    #: than refused. Only the artifact node has such a deployed state.
    normalises_legacy_rows: bool = False


#: The ``(node_type, label)`` shapes a PostgreSQL fleet may co-own.
#:
#: Two node types qualify: the governing constitution document, and a
#: Sovereign-signed reanchor artifact (#2893). Both are the same argument — the
#: node id IS the hash of the bytes, so every tenant computes identical
#: properties — and both are gated the same way: the writer must independently
#: own the underlying blob, so co-ownership follows possession of the content
#: rather than knowledge of a hash.
#:
#: This table is enforced in :meth:`AsyncGraphStore.add_node`, which is the only
#: writer of either shape today (inception and both reanchor writers go through
#: it). A future callsite that creates one of these rows through
#: :meth:`AsyncGraphStore.compare_and_swap_node` would bypass the table entirely
#: and has to carry the same guard, or it reopens the split this closes.
_SHARED_CONTENT_SHAPES = {
    # The anchor's agreement set is exactly what its predicate already pins, so
    # the check can never fire on its own here: two rows shareable under the
    # same node id necessarily agree on ``hash`` and ``type``. It is spelled out
    # rather than left off because the alternative is a default, and a default
    # is how the next shape added to this table ends up with no agreement check
    # without anyone having decided it should have none.
    ("document", "KESTREL_CONSTITUTION"): _SharedContentShape(
        is_shareable=_is_shareable_constitution_properties,
        identity_keys=frozenset({"hash", "type"}),
    ),
    (
        "constitution_amendment_artifact",
        "Signed Constitution Reanchor Artifact",
    ): _SharedContentShape(
        is_shareable=_is_shareable_amendment_artifact_properties,
        identity_keys=_SHAREABLE_ARTIFACT_KEYS,
        normalises_legacy_rows=True,
    ),
}


def _agrees_on_shared_identity(
    existing: Dict[str, Any],
    incoming: Dict[str, Any],
    shape: _SharedContentShape,
) -> bool:
    """Whether two co-owners of a shared row describe the same content.

    Compared over the shape's ``identity_keys`` rather than the whole property
    dict, because a wholesale comparison would refuse the constitution sharing
    that already works: each tenant stamps its own ``created_at`` when it first
    stores the document, and that difference is expected rather than a conflict.
    """
    return _canonical_shared_properties(
        existing, shape.identity_keys
    ) == _canonical_shared_properties(incoming, shape.identity_keys)


def _insert_owner_sql(db: AsyncDatabase, table: str, columns: str) -> str:
    """Return a backend-neutral insert-if-absent statement for an owner row."""
    placeholders = ", ".join("?" for _ in columns.split(","))
    if db.backend_type == "postgres":
        return (
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
            "ON CONFLICT DO NOTHING"
        )
    return f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({placeholders})"


async def record_graph_node_owner(
    db: AsyncDatabase, node_id: str, agent_id: str
) -> None:
    """Record authoritative ownership for one graph node.

    This low-level helper intentionally does not open a transaction so callers
    performing a larger import can keep the graph row and its ownership witness
    in the same transaction.  :class:`AsyncGraphStore` wraps its ordinary
    writes atomically.
    """
    if not node_id or not agent_id:
        raise ValueError("Graph node ownership requires node_id and agent_id")
    await db.execute(
        _insert_owner_sql(db, "graph_node_owners", "node_id, agent_id"),
        (node_id, agent_id),
    )


async def reserve_provisional_agent_owner(
    db: AsyncDatabase, agent_id: str
) -> None:
    """Reserve an agent root without claiming a conflicting shared-graph id.

    Avatar and backup storage can precede physical agent-root creation. Both
    bootstrap paths use this one guard while their surrounding transaction is
    open. The lock covers an absent PostgreSQL row as well as a present one, so
    the identity validation and ownership witness are one serialized action.
    """

    if not agent_id:
        raise ValueError("Provisional agent ownership requires an agent_id")
    await lock_graph_nodes_for_update(db, [agent_id], agent_id=agent_id)
    root = await db.fetchone(
        "SELECT node_type, properties FROM graph_nodes WHERE node_id = ?",
        (agent_id,),
    )
    if root and root[0] != "agent":
        raise ValueError("Agent owner id collides with a non-agent graph node")
    if root:
        try:
            properties = json.loads(root[1]) if root[1] else {}
        except (TypeError, ValueError) as exc:
            raise ValueError("Existing agent graph node has invalid properties") from exc
        declared = properties.get("agent_id")
        if declared and declared != agent_id:
            raise ValueError("Existing agent graph node declares another agent_id")

    owner_rows = await db.fetchall(
        "SELECT agent_id FROM graph_node_owners WHERE node_id = ?",
        (agent_id,),
    )
    if any(row[0] != agent_id for row in owner_rows):
        raise ValueError("Agent owner id is owned by another agent")
    await record_graph_node_owner(db, agent_id, agent_id)


async def record_graph_edge_owner(
    db: AsyncDatabase,
    source_id: str,
    target_id: str,
    label: str,
    agent_id: str,
) -> None:
    """Record authoritative ownership for one graph edge."""
    if not source_id or not target_id or not label or not agent_id:
        raise ValueError(
            "Graph edge ownership requires source_id, target_id, label, and agent_id"
        )
    await db.execute(
        _insert_owner_sql(
            db,
            "graph_edge_owners",
            "source_id, target_id, label, agent_id",
        ),
        (source_id, target_id, label, agent_id),
    )


async def release_graph_node_owners(
    db: AsyncDatabase,
    node_ids: List[str],
    agent_id: str,
) -> int:
    """Release one tenant's graph ownership without touching foreign lineage.

    Shared nodes retain their other ownership witnesses and edges.  An
    explicitly trusted cross-agent edge may intentionally reference a target
    outside its owner's private graph (including a target that is later
    removed), so foreign edge witnesses must survive this tenant's cleanup.
    Only edges that become completely ownerless are physically reclaimed.

    The caller owns the surrounding transaction so imports, purges, and
    ordinary deletes can compose this cleanup into their larger atomic unit.
    """
    if not agent_id:
        raise ValueError("Graph node ownership release requires an agent_id")

    unique_node_ids = await lock_graph_nodes_for_update(
        db, node_ids, agent_id=agent_id
    )
    affected = 0
    for start in range(0, len(unique_node_ids), _DELETE_ID_BATCH):
        batch = unique_node_ids[start:start + _DELETE_ID_BATCH]
        placeholders = ", ".join("?" for _ in batch)
        incident_params = tuple(batch) + tuple(batch)

        await db.execute(
            "DELETE FROM graph_edge_owners WHERE agent_id = ? AND ("
            f"source_id IN ({placeholders}) OR target_id IN ({placeholders}))",
            (agent_id,) + incident_params,
        )
        await db.execute(
            "DELETE FROM graph_node_owners WHERE agent_id = ? "
            f"AND node_id IN ({placeholders})",
            (agent_id,) + tuple(batch),
        )

        ownerless_rows = await db.fetchall(
            "SELECT node_id FROM graph_nodes "
            f"WHERE node_id IN ({placeholders}) AND NOT EXISTS ("
            "  SELECT 1 FROM graph_node_owners AS remaining_owner "
            "  WHERE remaining_owner.node_id = graph_nodes.node_id"
            ")",
            tuple(batch),
        )
        ownerless_ids = [row[0] for row in ownerless_rows]
        # Remove only physical edges whose final ownership witness belonged to
        # this tenant. Foreign trusted cross-agent references deliberately
        # survive even when the referenced node no longer exists locally.
        await db.execute(
            "DELETE FROM graph_edges WHERE ("
            f"source_id IN ({placeholders}) OR target_id IN ({placeholders})) "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM graph_edge_owners AS remaining_owner "
            "  WHERE remaining_owner.source_id = graph_edges.source_id "
            "  AND remaining_owner.target_id = graph_edges.target_id "
            "  AND remaining_owner.label = graph_edges.label"
            ")",
            incident_params,
        )

        if ownerless_ids:
            ownerless_placeholders = ", ".join("?" for _ in ownerless_ids)
            removed = await db.execute(
                f"DELETE FROM graph_nodes WHERE node_id IN ({ownerless_placeholders}) "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM graph_node_owners AS remaining_owner "
                "  WHERE remaining_owner.node_id = graph_nodes.node_id"
                ")",
                tuple(ownerless_ids),
            )
            if isinstance(removed, int):
                affected += removed
    return affected


async def _acquire_sqlite_graph_writer_slot(db: AsyncDatabase) -> None:
    """Acquire SQLite's writer slot without changing a graph row.

    The caller owns the surrounding transaction. This explicit write is also
    effective inside a same-task nested transaction, where a requested
    ``BEGIN IMMEDIATE`` cannot replace the outer deferred ``BEGIN``.
    """

    if db.backend_type == "sqlite":
        await db.execute("UPDATE graph_nodes SET node_id = node_id WHERE 0")


async def _lock_graph_node_ids_for_insert(
    db: AsyncDatabase,
    node_ids: Iterable[str],
) -> List[str]:
    """Reserve graph identifiers in canonical order, including absent rows.

    PostgreSQL row locks cannot cover a missing row.  The graph therefore has
    one transaction-advisory namespace shared by composed write-set locks and
    compare-and-create.  SQLite's writer slot provides the equivalent
    serialization.  This helper deliberately does not lock physical rows: a
    caller that has already established absence must not acquire a foreign
    tenant's row lock if another writer won just before the reservation.
    """

    unique_node_ids = sorted(
        dict.fromkeys(node_id for node_id in node_ids if node_id)
    )
    if db.backend_type == "sqlite":
        await _acquire_sqlite_graph_writer_slot(db)
        return unique_node_ids
    if db.backend_type != "postgres":
        return unique_node_ids
    if not unique_node_ids:
        return unique_node_ids

    if len(unique_node_ids) > _MAX_ABSENT_NODE_ADVISORY_LOCKS:
        raise ValueError(
            "A graph write may reserve at most "
            f"{_MAX_ABSENT_NODE_ADVISORY_LOCKS} absent node ids per transaction"
        )

    advisory_keys = sorted(
        {
            int.from_bytes(
                hashlib.sha256(
                    f"kestrel:graph-node:{node_id}".encode("utf-8")
                ).digest()[:8],
                "big",
                signed=True,
            )
            for node_id in unique_node_ids
        }
    )
    # The per-call cardinality check above is not enough: nested graph writes
    # join their caller's outer transaction, and transaction advisory locks do
    # not release when a successful savepoint exits. Count the locks already
    # retained by this PostgreSQL transaction and charge only genuinely new
    # bigint keys, so 129 singleton calls cannot bypass the same 128-lock cap.
    held_rows = await db.fetchall(
        "SELECT classid::bigint, objid::bigint, objsubid "
        "FROM pg_locks WHERE pid = pg_backend_pid() "
        "AND locktype = 'advisory' AND granted"
    )
    held_locks = {
        (int(row[0]), int(row[1]), int(row[2])) for row in held_rows
    }
    candidate_locks = set()
    for key in advisory_keys:
        unsigned_key = key & ((1 << 64) - 1)
        candidate_locks.add(
            (unsigned_key >> 32, unsigned_key & 0xFFFFFFFF, 1)
        )
    if len(held_locks | candidate_locks) > _MAX_ABSENT_NODE_ADVISORY_LOCKS:
        raise ValueError(
            "A graph write may reserve at most "
            f"{_MAX_ABSENT_NODE_ADVISORY_LOCKS} absent node ids per transaction"
        )
    for start in range(0, len(advisory_keys), _DELETE_ID_BATCH):
        batch = advisory_keys[start:start + _DELETE_ID_BATCH]
        values = ", ".join("(?::bigint)" for _ in batch)
        await db.fetchall(
            "SELECT pg_advisory_xact_lock(lock_key) "
            f"FROM (VALUES {values}) AS graph_locks(lock_key) "
            "ORDER BY lock_key",
            tuple(batch),
        )
    return unique_node_ids


async def _preflight_bound_graph_node_locks(
    db: AsyncDatabase,
    node_ids: Iterable[str],
    agent_id: str,
) -> None:
    """Reject rows outside a bound tenant before any physical lock is taken."""

    if db.backend_type != "postgres" or not agent_id:
        return
    unique_node_ids = sorted(
        dict.fromkeys(node_id for node_id in node_ids if node_id)
    )
    for start in range(0, len(unique_node_ids), _DELETE_ID_BATCH):
        batch = unique_node_ids[start:start + _DELETE_ID_BATCH]
        placeholders = ", ".join("?" for _ in batch)
        rows = await db.fetchall(
            "SELECT nodes.node_id, nodes.node_type, nodes.label, "
            "nodes.properties, EXISTS("
            "  SELECT 1 FROM graph_node_owners AS owner "
            "  WHERE owner.node_id = nodes.node_id AND owner.agent_id = ?"
            ") FROM graph_nodes AS nodes "
            f"WHERE nodes.node_id IN ({placeholders})",
            (agent_id, *batch),
        )
        for node_id, node_type, label, raw_properties, owned in rows:
            if owned:
                continue
            shape = _SHARED_CONTENT_SHAPES.get((node_type, label))
            try:
                properties = json.loads(raw_properties) if raw_properties else {}
            except (TypeError, ValueError):
                properties = {}
            if shape is not None and shape.is_shareable(properties, node_id):
                continue
            raise ValueError(
                "Cannot lock a graph node outside the bound agent"
            )


async def lock_graph_nodes_for_update(
    db: AsyncDatabase,
    node_ids: Iterable[str],
    *,
    agent_id: str = "",
) -> List[str]:
    """Lock graph rows in the canonical order before ownership mutation.

    Callers own the surrounding transaction. SQLite obtains serialization from
    that transaction's writer lock; PostgreSQL needs explicit row locks. When
    ``agent_id`` is supplied, physical locks are limited to that tenant's rows
    and valid fleet-shared content shapes. The sorted return value lets every
    caller process overlapping batches in the same order.
    """

    unique_node_ids = sorted(
        dict.fromkeys(node_id for node_id in node_ids if node_id)
    )
    if db.backend_type == "sqlite":
        await _acquire_sqlite_graph_writer_slot(db)
        return unique_node_ids
    if db.backend_type != "postgres":
        return unique_node_ids

    # Existing rows need only ordinary row locks. Taking one transaction-level
    # advisory lock for every row in a bulk purge/import exhausts PostgreSQL's
    # shared lock table on large tenant graphs because batching the statements
    # does not release locks before commit. Lock every existing row first, then
    # reserve only the identifiers that are genuinely absent. All callers use
    # that same row-before-absent order, while the absent helper enforces a hard
    # bound suitable for the small composed creation sets that need it.
    if agent_id:
        shared_clause = " OR ".join(
            "(graph_nodes.node_type = ? AND graph_nodes.label = ?)"
            for _ in _SHARED_CONTENT_SHAPES
        )
        shared_params = tuple(
            value for key in _SHARED_CONTENT_SHAPES for value in key
        )
        lock_scope = (
            "EXISTS(SELECT 1 FROM graph_node_owners AS lock_owner "
            "WHERE lock_owner.node_id = graph_nodes.node_id "
            "AND lock_owner.agent_id = ?)"
        )
        lock_scope_params: tuple[str, ...] = (agent_id,)
        if shared_clause:
            lock_scope = f"({lock_scope} OR {shared_clause})"
            lock_scope_params += shared_params
    else:
        lock_scope = "1 = 1"
        lock_scope_params = ()

    existing_ids: set[str] = set()
    for start in range(0, len(unique_node_ids), _DELETE_ID_BATCH):
        batch = unique_node_ids[start:start + _DELETE_ID_BATCH]
        placeholders = ", ".join("?" for _ in batch)
        rows = await db.fetchall(
            "SELECT node_id FROM graph_nodes "
            f"WHERE node_id IN ({placeholders}) "
            f"AND {lock_scope} "
            "ORDER BY node_id FOR UPDATE",
            (*batch, *lock_scope_params),
        )
        existing_ids.update(row[0] for row in rows)

    absent_ids = [
        node_id for node_id in unique_node_ids if node_id not in existing_ids
    ]
    await _lock_graph_node_ids_for_insert(db, absent_ids)
    if agent_id:
        # Recheck after waiting on an absent-id reservation. A foreign writer
        # may have won the identifier between the caller's cheap preflight and
        # this transaction's reservation, but it must never make us take that
        # tenant's row lock.
        await _preflight_bound_graph_node_locks(db, absent_ids, agent_id)

    # A creator may have committed while this transaction waited for an absent-
    # id reservation. Re-lock any such newly-present row while the reservation
    # is held so deletion/ownership mutation cannot race its physical row.
    for start in range(0, len(absent_ids), _DELETE_ID_BATCH):
        batch = absent_ids[start:start + _DELETE_ID_BATCH]
        placeholders = ", ".join("?" for _ in batch)
        await db.fetchall(
            "SELECT node_id FROM graph_nodes "
            f"WHERE node_id IN ({placeholders}) "
            f"AND {lock_scope} "
            "ORDER BY node_id FOR UPDATE",
            (*batch, *lock_scope_params),
        )
    return unique_node_ids


class NodeSwapResult(str, Enum):
    """Outcome of :meth:`AsyncGraphStore.compare_and_swap_node`.

    A ``str`` enum so the value compares equal to its plain-string form
    (``NodeSwapResult.SWAPPED == "swapped"``) and passes cleanly through
    the privacy wrapper and any JSON boundary a caller puts it behind.
    """
    SWAPPED = "swapped"
    PREDICATE_FAILED = "predicate_failed"
    NOT_FOUND = "not_found"
    # The caller constrained the swap/create to a set of node types
    # (``allowed_node_types``) and the *effective* type is outside it: on a
    # swap the **stored** row's ``node_type`` is not allowed (the row is left
    # untouched); on a compare-and-create the ``new_node.node_type`` is not
    # allowed (nothing is inserted). Only ever returned when a caller passes
    # ``allowed_node_types`` — the privacy wrapper uses it to fail a durable
    # graph CAS closed in volatile modes (#2672) without a TOCTOU pre-read.
    TYPE_NOT_ALLOWED = "type_not_allowed"


class NodeDeleteResult(str, Enum):
    """Outcome of :meth:`AsyncGraphStore.compare_and_delete_node`.

    ``DELETED`` means the node was removed from this store's tenant scope. A
    shared physical row may remain visible to its other owners after a bound
    store releases its own ownership witness.
    """

    DELETED = "deleted"
    PREDICATE_FAILED = "predicate_failed"
    NOT_FOUND = "not_found"


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    node_id: str
    node_type: str
    label: str
    properties: Dict[str, Any]


@dataclass
class Edge:
    """Represents an edge between nodes."""
    source_id: str
    target_id: str
    label: str
    properties: Optional[Dict[str, Any]] = None


class AsyncGraphStore:
    """Async knowledge graph storage.

    A store bound to ``agent_id`` is a tenant capability: every ordinary read,
    update, and delete is constrained by the ownership ledgers. Constructing a
    separate unbound store is the explicit privileged path used by migrations
    and single-database maintenance; a bound store has no per-call scope bypass.
    """

    def __init__(self, db: AsyncDatabase, agent_id: str = ""):
        self.db = db
        self.agent_id = agent_id

    def bind_agent(self, agent_id: str) -> None:
        """Bind subsequent graph writes to one authoritative agent owner.

        Rebinding a live store to a different agent would turn an isolation
        boundary into mutable ambient state, so it is rejected.  Construct a
        separate store for a separate agent instead.
        """
        if not agent_id:
            raise ValueError("Graph ownership binding requires a non-empty agent_id")
        if self.agent_id and self.agent_id != agent_id:
            raise ValueError("Graph store is already bound to a different agent")
        self.agent_id = agent_id

    async def lock_nodes_for_update(self, node_ids: Iterable[str]) -> List[str]:
        """Lock a complete multi-node write set in canonical order.

        The caller must own the surrounding transaction. Single-node writers
        lock internally, but a workflow that will touch several nodes must take
        the whole set first so two semantic write orders cannot become opposite
        PostgreSQL lock orders. SQLite uses the same call to acquire its writer
        slot before any read in the composed operation.
        """

        materialized = tuple(node_ids)
        await _preflight_bound_graph_node_locks(
            self.db, materialized, self.agent_id
        )
        return await lock_graph_nodes_for_update(
            self.db, materialized, agent_id=self.agent_id
        )

    def _node_owner(self, node: GraphNode) -> str:
        declared = node.properties.get("agent_id") if node.properties else None
        if node.node_type == "agent":
            if declared and declared != node.node_id:
                raise ValueError("Agent graph node declares a different agent_id")
            declared = node.node_id
        if declared is not None and not isinstance(declared, str):
            raise ValueError("Graph node properties.agent_id must be a string")
        if self.agent_id and declared and declared != self.agent_id:
            raise ValueError("Graph node owner does not match the bound agent")
        return self.agent_id or declared or ""

    def _node_scope(self, alias: str = "graph_nodes") -> tuple[str, tuple[str, ...]]:
        """Return the authoritative predicate for this store's node capability."""
        if not self.agent_id:
            return "1 = 1", ()
        return (
            "EXISTS (SELECT 1 FROM graph_node_owners AS node_scope_owner "
            f"WHERE node_scope_owner.node_id = {alias}.node_id "
            "AND node_scope_owner.agent_id = ?)",
            (self.agent_id,),
        )

    def _edge_scope(self, alias: str = "graph_edges") -> tuple[str, tuple[str, ...]]:
        """Return the authoritative predicate for this store's edge capability."""
        if not self.agent_id:
            return "1 = 1", ()
        return (
            "EXISTS (SELECT 1 FROM graph_edge_owners AS edge_scope_owner "
            f"WHERE edge_scope_owner.source_id = {alias}.source_id "
            f"AND edge_scope_owner.target_id = {alias}.target_id "
            f"AND edge_scope_owner.label = {alias}.label "
            "AND edge_scope_owner.agent_id = ?)",
            (self.agent_id,),
        )

    def _upsert_node_sql(self) -> str:
        """Get upsert SQL for nodes based on database backend."""
        if self.db.backend_type == "postgres":
            return """
                INSERT INTO graph_nodes (node_id, node_type, label, properties)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (node_id) DO UPDATE SET
                    node_type = EXCLUDED.node_type,
                    label = EXCLUDED.label,
                    properties = EXCLUDED.properties
            """
        return "INSERT OR REPLACE INTO graph_nodes (node_id, node_type, label, properties) VALUES (?, ?, ?, ?)"

    def _upsert_edge_sql(self) -> str:
        """Get upsert SQL for edges based on database backend."""
        if self.db.backend_type == "postgres":
            return """
                INSERT INTO graph_edges (source_id, target_id, label, properties)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (source_id, target_id, label) DO UPDATE SET
                    properties = EXCLUDED.properties
            """
        return "INSERT OR REPLACE INTO graph_edges (source_id, target_id, label, properties) VALUES (?, ?, ?, ?)"

    def _insert_if_absent_node_sql(self) -> str:
        """Backend-appropriate "insert only if node_id is still free" SQL.

        The dual of the upsert used by :meth:`add_node`: it inserts a brand-new
        row and does nothing (affecting zero rows) if the node already exists,
        which is exactly the atomic compare-and-create primitive that
        :meth:`compare_and_swap_node` needs for the ``expected is None`` case.
        """
        if self.db.backend_type == "postgres":
            return (
                "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (node_id) DO NOTHING"
            )
        return (
            "INSERT OR IGNORE INTO graph_nodes "
            "(node_id, node_type, label, properties) VALUES (?, ?, ?, ?)"
        )

    def _properties_match_predicate(self) -> str:
        """Backend WHERE fragment for "stored properties still equal the
        caller's snapshot", compared by JSON *value* rather than raw bytes.

        The trailing ``?`` binds the caller's snapshot re-serialized with
        ``json.dumps``. Both sides are normalized through the backend's JSON
        engine so the comparison is immune to representational drift that does
        not change the JSON value — which is exactly what a byte-exact
        ``properties = ?`` comparison got wrong:

        * ``properties IS NULL`` (or ``''``) is treated as ``{}`` — the same way
          :meth:`get_node` decodes such a row — so a caller who read ``{}`` back
          from a NULL/empty row can still swap it.
        * Minified vs. spaced serialization (``{"a":1}`` vs ``{"a": 1}``)
          compares equal, so a row persisted by any non-``add_node`` writer is
          still swappable against a ``get_node`` snapshot.

        SQLite uses ``json()`` (a minifying normalizer); Postgres uses ``jsonb``
        equality (order- and whitespace-independent). The predicate is always
        AND-ed with a ``node_id = ?`` primary-key match, so it only ever
        normalizes the single addressed row.
        """
        if self.db.backend_type == "postgres":
            return "COALESCE(NULLIF(properties, '')::jsonb, '{}'::jsonb) = ?::jsonb"
        return "json(COALESCE(NULLIF(properties, ''), '{}')) = json(?)"

    @staticmethod
    def _refuse_unshareable_properties(
        shape_key: tuple, properties: Dict[str, Any], node_id: str
    ) -> None:
        """Refuse a property set a fleet could not co-own.

        :meth:`add_node`'s check, and only its — which is the design rather
        than an oversight. A rule enforced at one of two writers is not a rule,
        so the *other* writer does not reimplement this one: it refuses
        fleet-shared rows outright (see :meth:`compare_and_swap_node`). One
        door knows the rules; the other declines to open.

        A no-op for every node type that is not content-addressed, which is the
        overwhelming majority.
        """
        shape = _SHARED_CONTENT_SHAPES.get(shape_key)
        if shape is None or shape.is_shareable(properties, node_id):
            return
        node_type, label = shape_key
        raise ValueError(
            "Cannot store a fleet-shared graph node whose properties fall "
            f"outside its shared shape: {node_type}/{label}"
        )

    async def add_node(self, node: GraphNode) -> None:
        """Add or update a node and its ownership witness atomically.

        The content write is a whole-row upsert (REPLACE / ON CONFLICT DO
        UPDATE) for the owning tenant: it clobbers that tenant's own prior
        row between a caller's read and this write. When you need "update X
        only if nobody changed it since I read it", use
        :meth:`compare_and_swap_node` instead — it closes the
        read-modify-write race that a hand-rolled retry loop around
        ``add_node`` can only ever narrow.

        For the handful of content-addressed shapes a fleet co-owns
        (``_SHARED_CONTENT_SHAPES``) the incoming node is validated *before*
        anything is read or written, whether or not a row already exists. The
        checks used to run only against an existing row, which made acceptance
        depend on insertion order: a property set outside the shared shape was
        stored happily for whichever agent got there first, and every sibling
        that presented the same content afterwards rolled back with an ownership
        error. Validating up front means such a node is refused for everyone or
        for no one — a loud failure at the first agent, in a command an operator
        is watching, instead of a silent fleet split discovered at the second.
        """
        owner = self._node_owner(node)
        shape = _SHARED_CONTENT_SHAPES.get((node.node_type, node.label))
        self._refuse_unshareable_properties(
            (node.node_type, node.label), node.properties, node.node_id
        )
        if self.db.backend_type == "postgres" and owner:
            # Establish whether this ID is already known to be outside the
            # tenant before taking either its advisory reservation or physical
            # row lock. Ordinary rows can never acquire a second owner, and a
            # caller presenting a shared shape can proceed only when the stored
            # identity is that same shared shape. Everything is re-read under
            # lock below; this cheap preflight rejects only writes that are
            # already guaranteed to fail and keeps an invalid tenant request
            # from queueing behind a foreign transaction.
            preflight_rows = await self.db.fetchall(
                "SELECT owners.agent_id, nodes.node_type, nodes.label "
                "FROM graph_node_owners AS owners "
                "LEFT JOIN graph_nodes AS nodes ON nodes.node_id = owners.node_id "
                "WHERE owners.node_id = ?",
                (node.node_id,),
            )
            preflight_owners = {row[0] for row in preflight_rows}
            has_foreign_owner = bool(preflight_owners - {owner})
            stored_shapes = {(row[1], row[2]) for row in preflight_rows}
            can_attempt_shared_admission = bool(
                shape is not None
                and stored_shapes == {(node.node_type, node.label)}
            )
            if has_foreign_owner and not can_attempt_shared_admission:
                raise ValueError(
                    "Cannot overwrite a graph node owned by another agent"
                )
        async with self.db.transaction():
            await lock_graph_nodes_for_update(
                self.db, [node.node_id], agent_id=owner
            )
            # A compatible fleet-shared row adds only an ownership witness.
            # Lock the physical row before deciding that on PostgreSQL: final-
            # owner deletion locks the same row first, so it cannot delete the
            # row between this identity read and ``record_graph_node_owner``.
            lock_suffix = (
                " FOR UPDATE" if self.db.backend_type == "postgres" else ""
            )
            existing = await self.db.fetchone(
                "SELECT node_type, label, properties FROM graph_nodes "
                f"WHERE node_id = ?{lock_suffix}",
                (node.node_id,),
            )
            existing_owner_rows = await self.db.fetchall(
                "SELECT agent_id FROM graph_node_owners WHERE node_id = ?",
                (node.node_id,),
            )
            existing_owners = {row[0] for row in existing_owner_rows}
            if existing and owner and not existing_owners:
                raise ValueError(
                    "Cannot claim or overwrite an unowned graph node"
                )
            foreign_owned = bool(owner and existing_owners and owner not in existing_owners)
            existing_properties = (
                json.loads(existing[2]) if existing and existing[2] else {}
            )
            owns_content_reference = False
            if (
                existing
                and owner
                and shape is not None
                and existing[0] == node.node_type
                and existing[1] == node.label
                and existing_properties.get("hash") == node.node_id
                and node.properties.get("hash") == node.node_id
            ):
                file_owner = await self.db.fetchone(
                    "SELECT 1 FROM file_owners "
                    "WHERE content_hash = ? AND agent_id = ?",
                    (node.node_id, owner),
                )
                owns_content_reference = file_owner is not None
            # A row this release would have written, or one the *previous*
            # release left behind carrying per-agent fields. The second is the
            # deployed state #2893 repairs, so it is admitted and normalised
            # rather than refused — see _is_normalisable_legacy_artifact.
            # ``node.properties`` is not re-checked in either clause below: it
            # was validated against the shape before this transaction opened, so
            # reaching here already means the incoming node is shareable.
            normalisable_legacy = bool(
                existing
                and shape is not None
                and shape.normalises_legacy_rows
                and existing[0] == node.node_type
                and existing[1] == node.label
                and owns_content_reference
                and _is_normalisable_legacy_artifact(
                    existing_properties, node.properties, node.node_id
                )
            )
            # Both halves are required. Shareability asks whether each row
            # *could* have been computed by any tenant; identity agreement asks
            # whether these two rows describe the same content. Without the
            # second, a node whose signed metadata disagrees with the stored row
            # still gains an owner — and because the stored row is deliberately
            # retained, this agent ends up owning a row that says something
            # other than what it just anchored, with no error to show for it.
            compatible_content_node = normalisable_legacy or bool(
                existing
                and shape is not None
                and existing[0] == node.node_type
                and existing[1] == node.label
                and shape.is_shareable(existing_properties, node.node_id)
                and _agrees_on_shared_identity(
                    existing_properties, node.properties, shape
                )
                and owns_content_reference
            )
            if (
                foreign_owned or len(existing_owners) > 1
            ) and not compatible_content_node:
                raise ValueError(
                    "Cannot overwrite a graph node owned by another agent"
                )
            # The shape has to be read off the *stored* row as well as off the
            # incoming node. ``add_node`` is a whole-row upsert — it writes
            # ``node_type`` and ``label`` too — so deriving the guards only from
            # what the caller declared let a sole owner walk around every rule
            # below simply by relabelling: present the artifact's node id as an
            # ``episode``, ``shape`` comes back ``None``, no check runs, and the
            # upsert replaces the fleet's governance row wholesale. The next
            # agent to anchor the genuine artifact then meets a row it cannot
            # match and rolls back — the same split, through the same door I had
            # just closed on the swap path and not here.
            #
            # Placed after the ownership refusal on purpose: a foreign caller
            # must keep getting "owned by another agent" and learn nothing more
            # about a row it cannot see.
            stored_key = (existing[0], existing[1]) if existing else None
            if (
                stored_key is not None
                and stored_key in _SHARED_CONTENT_SHAPES
                and stored_key != (node.node_type, node.label)
            ):
                raise ValueError(
                    "Cannot change the node_type or label of a fleet-shared "
                    f"graph node: {stored_key[0]}/{stored_key[1]}"
                )
            # Being the only owner today is not a licence to redefine what the
            # row says. These node ids ARE the hash of the bytes, so the
            # identity fields cannot legitimately change while the id stays the
            # same — a different artifact is a different node. Allowing a sole
            # owner to swap in another well-formed ``signer`` looked harmless
            # (nobody else is on the row yet) and is not: every sibling that
            # later anchors the genuine file now disagrees with the stored row
            # and rolls back, with no way to repair it. That is this issue's
            # fleet split, reachable by one agent acting alone.
            if (
                existing
                and shape is not None
                and not normalisable_legacy
                and not _agrees_on_shared_identity(
                    existing_properties, node.properties, shape
                )
            ):
                raise ValueError(
                    "Cannot change the content-derived identity of a "
                    f"fleet-shared graph node: {node.node_type}/{node.label}"
                )

            # For an identical shared node, retain the canonical row bytes and
            # add only the second ownership witness.  This avoids one tenant
            # rewriting another tenant's content-addressed row serialization.
            # ...except when the stored row is a legacy artifact carrying the
            # previous release's per-agent fields. Then the write is the
            # normalisation itself: it strips those fields and leaves only what
            # the artifact bytes fix, which is the whole point of sharing the
            # row. Nothing content-identifying changes, so this is not one
            # tenant rewriting another's content — it is the row becoming what
            # both tenants independently compute.
            current_owner_can_update = bool(
                not existing or not owner or existing_owners == {owner}
            ) or normalisable_legacy
            if current_owner_can_update:
                await self.db.execute(
                    self._upsert_node_sql(),
                    (
                        node.node_id,
                        node.node_type,
                        node.label,
                        json.dumps(node.properties),
                    ),
                )
            if owner:
                await record_graph_node_owner(self.db, node.node_id, owner)

    async def compare_and_swap_node(
        self,
        node_id: str,
        expected: Optional[Dict[str, Any]],
        new_node: GraphNode,
        allowed_node_types: Optional[frozenset] = None,
        *,
        expected_node_type: Optional[str] = None,
        expected_label: Optional[str] = None,
    ) -> NodeSwapResult:
        """Atomically update a node's ``properties`` only if they still match.

        This is the race-free conditional-update primitive for the knowledge
        graph. Unlike :meth:`add_node` (a whole-row clobber), the check and the
        write happen as one serialized unit at the storage layer, so no
        concurrent writer can slip a change in between — closing the TOCTOU
        window that a read-then-``add_node`` retry loop can only narrow.

        The primitive is deliberately **properties-only**: it compares — and,
        for an existing node, writes — the ``properties`` column alone. A node's
        ``node_type`` and ``label`` are set once at creation and are *not*
        touched by a swap (``new_node.node_type`` / ``new_node.label`` are
        ignored on the swap path; they are used only when compare-and-create
        inserts a brand-new row). This alignment — predicate, write, and the
        ``expected`` snapshot all on ``properties`` — is also why a swap cannot
        clobber a concurrent ``node_type`` / ``label`` change: it never writes
        those columns. A callsite that also needs to change ``node_type`` /
        ``label`` uses :meth:`add_node`, or gates on properties here and lets the
        type/label ride along in the properties it swaps.

        Args:
            node_id: The node to conditionally update.
            expected: The ``properties`` snapshot the caller last read (exactly
                what :meth:`get_node` returned for this node's ``properties``).
                The swap succeeds only if the row's stored ``properties`` still
                decode to the same JSON *value* as this snapshot — i.e. no writer
                has changed the node's properties since the read. Pass ``None``
                to mean "I read no node": the swap then acts as
                compare-and-create, succeeding only while the node is still
                absent.
            new_node: The replacement state. On a swap only ``new_node.properties``
                is written; on a compare-and-create (``expected is None``) the
                full node — ``node_type``, ``label`` and ``properties`` — is
                inserted. The row identity always stays ``node_id``.
            allowed_node_types: Optional set constraining the *effective*
                ``node_type`` the operation may touch. When given, the swap
                lands only if the **stored** row's ``node_type`` is in the set
                (added to the ``UPDATE`` predicate, so a disallowed row matches
                zero rows and is never rewritten), and a compare-and-create
                lands only if ``new_node.node_type`` is in the set. A blocked
                operation returns :attr:`NodeSwapResult.TYPE_NOT_ALLOWED`
                without writing. This is the atomic, TOCTOU-free hook the
                privacy wrapper uses to keep a durable graph CAS from silently
                rewriting a user-derived node under a spoofed structural
                ``new_node.node_type`` in volatile modes (#2672) — because a
                swap ignores ``new_node.node_type`` and writes ``properties``
                onto whatever row already exists. ``None`` (the default) imposes
                no type constraint, preserving the primitive's original
                behaviour for every non-privacy caller.
            expected_node_type: Optional exact stored ``node_type`` predicate.
                Must be supplied together with ``expected_label``. The pair is
                added to the same atomic ``UPDATE`` predicate as ``expected``.
            expected_label: Optional exact stored ``label`` predicate. Must be
                supplied together with ``expected_node_type``.

        Returns:
            * :attr:`NodeSwapResult.SWAPPED` — the predicate held and the write
              landed.
            * :attr:`NodeSwapResult.PREDICATE_FAILED` — the node exists *and is
              visible to this caller* but its stored ``properties`` no longer
              match ``expected`` (a concurrent writer won), or
              ``expected is None`` and this caller already owns a node at
              ``node_id`` (a genuine create conflict). The existing row —
              including whatever the other writer wrote — is left untouched.
            * :attr:`NodeSwapResult.NOT_FOUND` — the row is absent *from this
              caller's scope*: either genuinely absent, or (on a bound store)
              owned by another tenant and therefore invisible here. Returned both
              when ``expected`` was a snapshot of a now-missing row and when
              ``expected is None`` conflicted with a foreign-owned id — so a
              bound tenant cannot distinguish "absent" from "another tenant owns
              it", matching :meth:`get_node`.

        Note:
            Equality is on the *JSON value* of ``properties``, not the raw stored
            bytes. Both the stored ``properties`` and ``expected`` are normalized
            through the backend's JSON engine (SQLite ``json()`` / Postgres
            ``jsonb``), so a swap is accepted whenever no writer changed the
            value — even if the row was persisted with different-but-equivalent
            JSON text (minified vs. spaced) or with a ``NULL``/empty
            ``properties`` column that :meth:`get_node` decodes to ``{}``. A
            byte-exact comparison rejected both of those (they are valid,
            unchanged rows) — see :meth:`_properties_match_predicate`. It stays
            fail-closed on a real ``properties`` change: a writer that touched
            ``properties`` since your read yields ``PREDICATE_FAILED`` rather
            than overwriting their change. A concurrent ``node_type`` / ``label``
            change is neither detected nor clobbered by the default
            properties-only signature. Callers that own an exact graph shape can
            pass ``expected_node_type`` and ``expected_label`` to widen only the
            predicate while retaining the properties-only write.

        Tenant scoping:
            On a store bound to an agent (:meth:`bind_agent`) this primitive is
            ownership-scoped exactly like :meth:`add_node` and :meth:`get_node`.
            The swap ``UPDATE`` and its failure-classification read run through
            this store's ownership predicate, so a bound agent can never swap a
            node owned by another tenant (the write matches zero rows and the
            node reads back as ``NOT_FOUND``, not ``PREDICATE_FAILED``). The same
            holds for compare-and-create: an insert that loses to a foreign-owned
            ``node_id`` re-reads under this store's scope and reports
            ``NOT_FOUND`` (the row is invisible here), never ``PREDICATE_FAILED``
            — so the create path leaks no more existence than :meth:`get_node`.
            A successful compare-and-create records the caller's ownership
            witness in the same transaction, so the freshly-created row is
            visible to its creator on the next scoped read. An unbound store (no
            ``agent_id``) records no ownership and scopes to ``1 = 1`` — the
            ownerless behaviour the primitive shipped with, where any existing
            row is a visible ``PREDICATE_FAILED`` conflict.
        """
        identity_clause, identity_params = self._identity_predicate(
            expected_node_type=expected_node_type,
            expected_label=expected_label,
        )
        if identity_params and (
            new_node.node_type != expected_node_type
            or new_node.label != expected_label
        ):
            raise ValueError(
                "new_node identity must match expected_node_type and expected_label"
            )
        new_properties = json.dumps(new_node.properties)
        # Resolve the caller's authoritative owner up front, exactly like
        # add_node: a bound store may only write nodes it owns, and rejects a
        # new_node that declares a foreign agent_id. For an unbound store this
        # is "" and every scope below collapses to ``1 = 1`` — preserving the
        # ownerless CAS semantics the primitive shipped with.
        owner = self._node_owner(new_node)
        scope, scope_params = self._node_scope()

        # One serialized write unit: the check and the write commit or roll back
        # together, and the failure-classification read sees the same committed
        # snapshot. SQLite serializes this under its per-connection write lock;
        # Postgres runs the conditional UPDATE under a row lock so concurrent
        # swaps block and re-evaluate the predicate against the committed row.
        # Build the optional "stored/created type must be allowed" predicate
        # once. For a swap it is AND-ed into the UPDATE so a disallowed stored
        # row matches zero rows (never rewritten); for a create the new node's
        # own type is checked before insert. Kept as a set membership so the
        # SQL uses a parameterized ``IN`` list — no interpolated values.
        type_clause = ""
        type_params: tuple = ()
        if allowed_node_types is not None:
            allowed_tuple = tuple(allowed_node_types)
            if allowed_tuple:
                placeholders = ", ".join("?" for _ in allowed_tuple)
                type_clause = f" AND node_type IN ({placeholders})"
                type_params = allowed_tuple
            else:
                type_clause = " AND 1 = 0"

        async with self.db.transaction():
            if expected is None:
                # Compare-and-create: only lands while the node is still absent.
                # A create writes ``new_node.node_type`` verbatim, so gate on it
                # directly and insert nothing when it is disallowed.
                if (
                    allowed_node_types is not None
                    and new_node.node_type not in allowed_node_types
                ):
                    return NodeSwapResult.TYPE_NOT_ALLOWED
                # A create writes ``new_node``'s shape verbatim, and unlike a
                # swap there is nothing to conflict with: no stored row, so no
                # identity to preserve, no co-owner to overrule, and no read
                # needed to know that. Only shareability applies, which is a
                # question about the incoming properties alone — the same check
                # ``add_node`` makes, and the reason a create stays allowed
                # here while a swap does not. The privacy wrapper admits
                # ``document``/``KESTREL_CONSTITUTION`` as a content-free
                # structural type and creates it through this path (#2672).
                #
                # Validated against ``node_id`` — the row identity this
                # primitive writes — not ``new_node.node_id``, which it ignores.
                self._refuse_unshareable_properties(
                    (new_node.node_type, new_node.label), new_node.properties, node_id
                )
                if self.db.backend_type == "postgres" and owner:
                    # Refuse an already-present foreign id before taking this
                    # graph namespace's advisory lock.  The scoped result is
                    # unchanged (NOT_FOUND), but one tenant cannot make an
                    # invalid create queue behind another tenant's row work.
                    visibility = await self.db.fetchone(
                        "SELECT "
                        "EXISTS(SELECT 1 FROM graph_nodes WHERE node_id = ?), "
                        "EXISTS(SELECT 1 FROM graph_node_owners "
                        "       WHERE node_id = ? AND agent_id = ?)",
                        (node_id, node_id, owner),
                    )
                    if visibility and visibility[0]:
                        return (
                            NodeSwapResult.PREDICATE_FAILED
                            if visibility[1]
                            else NodeSwapResult.NOT_FOUND
                        )
                # Share the absent-row reservation used by composed graph
                # writers.  Without this, PostgreSQL's conflict-safe INSERT
                # can land while another transaction believes its complete
                # (currently absent) graph write set is exclusively reserved.
                await _lock_graph_node_ids_for_insert(self.db, [node_id])
                affected = await self.db.execute(
                    self._insert_if_absent_node_sql(),
                    (node_id, new_node.node_type, new_node.label, new_properties),
                )
                if _rows_affected(affected) > 0:
                    # Record ownership inside the same transaction so a bound
                    # creator can actually read its new node back through the
                    # scoped predicate; without this the row is orphaned and
                    # invisible to the very agent that created it.
                    if owner:
                        await record_graph_node_owner(self.db, node_id, owner)
                    return NodeSwapResult.SWAPPED
                # The insert matched no row: the ``node_id`` primary key is
                # already taken. Classify with the SAME ownership scope the swap
                # path uses (below), not an unscoped "it exists": ``INSERT OR
                # IGNORE`` / ``ON CONFLICT DO NOTHING`` gates on the global PK, so
                # a bare "0 rows -> PREDICATE_FAILED" would let a bound tenant
                # tell a *foreign-owned* id (invisible to it under get_node) apart
                # from an absent one — an existence leak across the tenant
                # boundary. Re-read under this store's scope so a foreign node
                # reads back as NOT_FOUND exactly like get_node, and only a row
                # this caller can actually see (its own prior create) is a real
                # PREDICATE_FAILED conflict.
                exists = await self.db.fetchone(
                    f"SELECT 1 FROM graph_nodes WHERE node_id = ? AND {scope}",
                    (node_id, *scope_params),
                )
                if exists is not None:
                    return NodeSwapResult.PREDICATE_FAILED
                return NodeSwapResult.NOT_FOUND

            # An existing fleet-shared row has exactly ONE writer: ``add_node``.
            # This primitive refuses them rather than reimplementing that door's
            # rules — shareability, immutable identity, co-ownership — a second
            # time against a different write path. A *create* is different and
            # is allowed above: there is no stored row to preserve, so only
            # shareability applies.
            #
            # Refused by a clause in the conditional UPDATE rather than by
            # inspecting the row first, which matters for two reasons. A check
            # before a write is not a check: a concurrent ``add_node`` can retype
            # the row in between. And reading first would make this method
            # read-then-write, which it has never been — SQLite opens a deferred
            # transaction and its write lock is per-connection, so a second
            # connection committing in between leaves a snapshot that cannot be
            # upgraded, and a losing swap would raise instead of returning
            # PREDICATE_FAILED. That contract belongs to every node type here,
            # not just these two, and must not be spent on them.
            #
            # The classification read below reports the refusal as
            # TYPE_NOT_ALLOWED — already this method's answer for "a type this
            # operation may not touch", and already what the privacy wrapper
            # converts into a PrivacyViolationError.
            shared_clause = "".join(
                " AND NOT (node_type = ? AND label = ?)"
                for _ in _SHARED_CONTENT_SHAPES
            )
            shared_params = tuple(
                value for key in _SHARED_CONTENT_SHAPES for value in key
            )

            expected_properties = json.dumps(expected)
            # Properties-only: the SET touches the same single column the
            # predicate gates on, so a concurrent node_type/label change is
            # never overwritten (we don't write those columns). The scope
            # predicate binds the write to this store's ownership capability so
            # a bound agent can never swap a node it does not own — the UPDATE
            # simply matches zero rows, exactly like every other bound write.
            affected = await self.db.execute(
                "UPDATE graph_nodes "
                "SET properties = ? "
                f"WHERE node_id = ? AND {self._properties_match_predicate()}"
                f"{identity_clause}{type_clause}{shared_clause} "
                f"AND {scope}",
                (
                    new_properties,
                    node_id,
                    expected_properties,
                    *identity_params,
                    *type_params,
                    *shared_params,
                    *scope_params,
                ),
            )
            if _rows_affected(affected) > 0:
                return NodeSwapResult.SWAPPED

            # Zero rows changed: distinguish "predicate no longer holds", "type
            # not allowed", and "row genuinely absent" with one read in the same
            # serialized section. The read carries the SAME ownership scope, so a
            # foreign-owned node reads as NOT_FOUND (invisible to this tenant)
            # rather than leaking its existence. When the caller passed
            # ``allowed_node_types``, a visible row whose stored ``node_type`` is
            # outside the set is TYPE_NOT_ALLOWED (a policy block, not a lost
            # race) — this is how the wrapper tells "someone else changed it"
            # apart from "this is a user-derived row I must not rewrite".
            existing = await self.db.fetchone(
                "SELECT node_type, label FROM graph_nodes "
                f"WHERE node_id = ? AND {scope}",
                (node_id, *scope_params),
            )
            if existing is None:
                return NodeSwapResult.NOT_FOUND
            # A fleet-shared row is a type this primitive may not touch, which
            # is what TYPE_NOT_ALLOWED already means: a policy block rather than
            # a lost race, and the answer the privacy wrapper knows how to
            # convert. Reported from the read that was already here, so the
            # refusal costs no extra query and races nothing — the UPDATE has
            # already declined to write.
            if (existing[0], existing[1]) in _SHARED_CONTENT_SHAPES:
                return NodeSwapResult.TYPE_NOT_ALLOWED
            if (
                allowed_node_types is not None
                and existing[0] not in allowed_node_types
            ):
                return NodeSwapResult.TYPE_NOT_ALLOWED
            return NodeSwapResult.PREDICATE_FAILED

    @staticmethod
    def _identity_predicate(
        *,
        expected_node_type: Optional[str],
        expected_label: Optional[str],
    ) -> tuple[str, tuple[str, ...]]:
        """Build one exact type+label predicate or reject a partial identity."""

        if (expected_node_type is None) != (expected_label is None):
            raise ValueError(
                "expected_node_type and expected_label must be supplied together"
            )
        if expected_node_type is None:
            return "", ()
        return (
            " AND node_type = ? AND label = ?",
            (expected_node_type, expected_label),
        )

    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        scope, scope_params = self._node_scope()
        row = await self.db.fetchone(
            "SELECT node_id, node_type, label, properties FROM graph_nodes "
            f"WHERE node_id = ? AND {scope}",
            (node_id,) + scope_params,
        )
        if not row:
            return None
        return GraphNode(
            node_id=row[0],
            node_type=row[1],
            label=row[2],
            properties=json.loads(row[3]) if row[3] else {}
        )
    
    async def get_nodes_by_type(self, node_type: str) -> List[GraphNode]:
        """Get all nodes of a specific type."""
        scope, scope_params = self._node_scope()
        rows = await self.db.fetchall(
            "SELECT node_id, node_type, label, properties FROM graph_nodes "
            f"WHERE node_type = ? AND {scope}",
            (node_type,) + scope_params,
        )
        return [
            GraphNode(
                node_id=row[0],
                node_type=row[1],
                label=row[2],
                properties=json.loads(row[3]) if row[3] else {}
            )
            for row in rows
        ]

    # -----------------------------------------------------------------
    # Property-level query helpers (use JSON-path indexes)
    # -----------------------------------------------------------------

    def _json_extract(self, column: str, path: str) -> str:
        """Return backend-appropriate JSON extraction SQL.

        SQLite:     json_extract(column, '$.path')
        PostgreSQL: (column::jsonb)->>'path'
        """
        if self.db.backend_type == "postgres":
            return f"({column}::jsonb)->>'{path}'"
        return f"json_extract({column}, '$.{path}')"

    async def query_nodes_by_type_and_property(
        self,
        node_type: str,
        filters: Optional[Dict[str, Any]] = None,
        *,
        created_since: Optional[str] = None,
        order_by_created: bool = True,
        limit: int = 200,
    ) -> List[GraphNode]:
        """Query graph nodes by type with property-level SQL filters.

        Pushes equality and range filters into SQL so the database can
        use the JSON-path partial indexes (``idx_graph_nodes_agent``,
        ``idx_graph_nodes_action_status``, ``idx_graph_nodes_action_created``).

        Args:
            node_type: Required ``node_type`` value (e.g. ``"action_item"``).
            filters: Dict of ``{property_name: value}`` for equality checks
                pushed into ``WHERE json_extract(properties, '$.key') = ?``.
            created_since: ISO-8601 timestamp lower bound on
                ``properties->>'created_at'``.  Uses ``>=`` comparison.
            order_by_created: If True (default), results are ordered by
                ``properties->>'created_at' DESC``.
            limit: Maximum rows returned (clamped to 1-10000).

        Returns:
            List of matching :class:`GraphNode` instances.
        """
        limit = max(1, min(limit, 10000))
        clauses: List[str] = ["node_type = ?"]
        params: List[Any] = [node_type]

        scope, scope_params = self._node_scope()
        clauses.append(scope)
        params.extend(scope_params)

        for key, value in (filters or {}).items():
            clauses.append(f"{self._json_extract('properties', key)} = ?")
            params.append(value)

        if created_since is not None:
            clauses.append(f"{self._json_extract('properties', 'created_at')} >= ?")
            params.append(created_since)

        where = " AND ".join(clauses)
        order = ""
        if order_by_created:
            order = f" ORDER BY {self._json_extract('properties', 'created_at')} DESC"

        sql = (
            f"SELECT node_id, node_type, label, properties "
            f"FROM graph_nodes WHERE {where}{order} LIMIT ?"
        )
        params.append(limit)

        rows = await self.db.fetchall(sql, tuple(params))
        return [
            GraphNode(
                node_id=row[0],
                node_type=row[1],
                label=row[2],
                properties=json.loads(row[3]) if row[3] else {},
            )
            for row in rows
        ]

    async def _delete_node_in_transaction(self, node_id: str) -> bool:
        """Delete/release one node while the caller holds a transaction."""

        if self.agent_id:
            owned = await self.db.fetchone(
                "SELECT 1 FROM graph_node_owners "
                "WHERE node_id = ? AND agent_id = ?",
                (node_id, self.agent_id),
            )
            if not owned:
                return False

            await release_graph_node_owners(self.db, [node_id], self.agent_id)
            return True

        await self.db.execute(
            "DELETE FROM graph_edge_owners "
            "WHERE source_id = ? OR target_id = ?",
            (node_id, node_id),
        )
        await self.db.execute(
            "DELETE FROM graph_edges WHERE source_id = ? OR target_id = ?",
            (node_id, node_id),
        )
        await self.db.execute(
            "DELETE FROM graph_node_owners WHERE node_id = ?",
            (node_id,),
        )
        removed = await self.db.execute(
            "DELETE FROM graph_nodes WHERE node_id = ?",
            (node_id,),
        )
        return bool(_rows_affected(removed))

    async def _visible_node_identity_for_update(
        self, node_id: str
    ) -> Optional[tuple[str, str]]:
        """Read and lock one tenant-visible graph identity inside a transaction."""

        scope, scope_params = self._node_scope()
        if self.agent_id:
            # Scope before taking either the PostgreSQL advisory lock or the
            # physical row lock. An invisible id must behave exactly like an
            # absent id and must not let one tenant hold another tenant's locks
            # until a caller-managed outer transaction eventually commits.
            visible = await self.db.fetchone(
                "SELECT 1 FROM graph_node_owners "
                "WHERE node_id = ? AND agent_id = ?",
                (node_id, self.agent_id),
            )
            if visible is None:
                return None
        # ``transaction(immediate=True)`` cannot upgrade a same-task outer
        # deferred transaction because SQLite's nested transaction scope is a
        # no-op. Acquire explicitly before reading identity so the composed
        # public ``AsyncStorage.transaction()`` path remains serialized too.
        # The scoped read below is deliberately repeated after locking: an
        # ownership witness can disappear between the cheap probe and the lock.
        await lock_graph_nodes_for_update(
            self.db, [node_id], agent_id=self.agent_id
        )
        return await self.db.fetchone(
            "SELECT node_type, label FROM graph_nodes "
            f"WHERE node_id = ? AND {scope}",
            (node_id, *scope_params),
        )

    async def delete_node(self, node_id: str) -> None:
        """Release this store's node witness and reclaim ownerless rows.

        A bound delete never removes another tenant's witness. Shared physical
        nodes and edges remain until their final owner releases them. An
        unbound maintenance store preserves the legacy physical-delete
        behavior.
        """
        async with self.db.transaction(immediate=True):
            # Take the graph-row lock first when the row exists, but always run
            # cleanup. Ownership and edge ledgers intentionally have no foreign
            # keys, so an interrupted/legacy write can leave repairable records
            # after the physical node has already disappeared.
            await self._visible_node_identity_for_update(node_id)
            await self._delete_node_in_transaction(node_id)

    async def compare_and_delete_node(
        self,
        node_id: str,
        *,
        expected_node_type: str,
        expected_label: str,
    ) -> NodeDeleteResult:
        """Delete only while the visible node's exact identity still matches.

        The type+label read and deletion share one serialized transaction.
        SQLite acquires its writer slot before reading; PostgreSQL locks the
        selected graph row. A whole-row writer that replaces the node before
        this operation therefore yields ``PREDICATE_FAILED`` without losing its
        replacement, while a writer arriving afterward waits until deletion
        commits.

        On a bound store, ``DELETED`` means this tenant's ownership witness was
        released. A shared physical row remains for any other owners.
        """

        async with self.db.transaction(immediate=True):
            existing = await self._visible_node_identity_for_update(node_id)
            if existing is None:
                return NodeDeleteResult.NOT_FOUND
            if existing != (expected_node_type, expected_label):
                return NodeDeleteResult.PREDICATE_FAILED
            deleted = await self._delete_node_in_transaction(node_id)
            return (
                NodeDeleteResult.DELETED
                if deleted
                else NodeDeleteResult.NOT_FOUND
            )

    async def purge_agent_nodes(
        self, agent_id: str, *, since_iso: Optional[str] = None
    ) -> int:
        """Release graph rows authoritatively owned by ``agent_id`` (#767/#867).

        EPHEMERAL agents are not supposed to write to ``graph_nodes`` —
        the privacy wrapper rejects persistent writes in that mode. This
        method exists as the safety net for the case where a write
        slipped through anyway.

        Ownership is selected from ``graph_node_owners``. Shared physical rows
        survive with their other ownership witnesses; ownerless nodes and
        edges are reclaimed. A bound store may purge only its own agent.

        Args:
            agent_id: agent's DID.
            since_iso: Optional ISO-8601 timestamp.  When provided, only
                nodes whose ``properties.created_at >= since_iso`` are
                destroyed — this scopes the EPHEMERAL leak-purge to the
                rows authored *during* the EPHEMERAL stint and leaves
                preexisting NORMAL data alone (#867).  When omitted,
                every node owned by this agent is destroyed (legacy
                behaviour preserved for restore-from-CAR and explicit
                administrative wipes).

        Returns:
            Number of node rows destroyed. Zero is the happy path; any
            non-zero value during a leak-purge means the privacy layer
            leaked.
        """
        if not agent_id:
            return 0
        if self.agent_id and self.agent_id != agent_id:
            raise ValueError("A bound graph store cannot purge another agent")

        if self.db.backend_type == "postgres":
            # graph_nodes.properties.created_at is documented as
            # ``YYYY-MM-DDTHH:MM:SS+00:00`` (ISO with T separator, fixed
            # offset).  Normalise it to ``YYYY-MM-DD HH:MM:SS`` so it can
            # be lex-compared against the SQLite-format watermark the
            # privacy wrapper records.  Without this normalisation the
            # ``T`` (0x54) sorts AFTER space (0x20) and every same-day
            # graph row appears strictly greater than the watermark — so
            # pre-stint nodes get purged.
            created_normalized = (
                "to_char(("
                "  CASE WHEN (properties::jsonb->>'created_at') IS NULL THEN NULL "
                "       ELSE ((properties::jsonb->>'created_at')::timestamptz) "
                "  END "
                ") AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')"
            )
        else:
            # SQLite normalisation: ``T`` → space, then truncate to
            # ``YYYY-MM-DD HH:MM:SS`` (length 19).  Handles both ISO
            # (``2026-04-26T16:31:06+00:00``) and SQLite-format (``2026-04-26
            # 16:31:06``) inputs uniformly.
            created_normalized = (
                "substr("
                "  replace(json_extract(properties, '$.created_at'), 'T', ' '), "
                "  1, 19"
                ")"
            )

        ownership_clause = (
            "EXISTS (SELECT 1 FROM graph_node_owners AS purge_owner "
            "WHERE purge_owner.node_id = graph_nodes.node_id "
            "AND purge_owner.agent_id = ?)"
        )
        if since_iso:
            # Nodes without a ``created_at`` are excluded from the scoped
            # purge — we can't prove they're in-window leaks, so we
            # preserve them rather than risk destroying real preexisting
            # data.  Operators get a WARNING below if any such nodes
            # exist for this agent (visible-but-skipped surface).
            agent_clause = (
                f"({ownership_clause} AND {created_normalized} IS NOT NULL "
                f"AND {created_normalized} >= ?)"
            )
            agent_args: tuple = (agent_id, since_iso)
        else:
            agent_clause = ownership_clause
            agent_args = (agent_id,)

        # When scoping by since_iso, count nodes for this agent that have
        # NO created_at — they're skipped by the predicate and we want
        # operators to see them so they can investigate the missing
        # provenance.  Cheap row count, scoped to the agent.
        if since_iso:
            try:
                untimed_row = await self.db.fetchone(
                    f"SELECT COUNT(*) FROM graph_nodes "
                    f"WHERE {ownership_clause} "
                    f"  AND {created_normalized} IS NULL",
                    (agent_id,),
                )
                untimed = int(untimed_row[0]) if untimed_row else 0
                if untimed > 0:
                    logger.warning(
                        "purge_agent_nodes (scoped): %d node(s) for agent=%s "
                        "have no properties.created_at and were skipped — "
                        "leak coverage is incomplete for them.  Investigate "
                        "the writer and stamp created_at going forward.",
                        untimed, agent_id,
                    )
            except Exception:
                # Pre-flight count is informational only.
                pass

        async with self.db.transaction():
            selected = await self.db.fetchall(
                f"SELECT node_id FROM graph_nodes WHERE {agent_clause}",
                agent_args,
            )
            node_ids = [row[0] for row in selected]
            return await release_graph_node_owners(self.db, node_ids, agent_id)
    
    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        label: str,
        properties: Optional[Dict] = None,
    ) -> None:
        """Add an edge between nodes.

        Upserts by (source_id, target_id, label) — calling add_edge twice
        with the same triple updates the properties, not duplicates the edge.
        A tenant-bound writer must own both endpoints; intentional lineage
        edges to an external parent use :meth:`add_trusted_cross_agent_edge`.
        """
        await self._add_edge(
            source_id,
            target_id,
            label,
            properties,
            trusted_cross_agent=False,
        )

    async def add_trusted_cross_agent_edge(
        self,
        source_id: str,
        target_id: str,
        label: str,
        properties: Optional[Dict] = None,
    ) -> None:
        """Add an intentionally cross-agent edge owned by this bound agent.

        This is a narrow infrastructure writer for relationships such as a
        child's ``spawned_by`` edge when the parent node lives in another
        tenant database (or is owned only by the parent in a shared database).
        The source must be owned by the bound agent; unbound callers and
        arbitrary foreign-source writes are rejected.
        """
        if not self.agent_id:
            raise ValueError("Trusted cross-agent edges require a bound graph store")
        await self._add_edge(
            source_id,
            target_id,
            label,
            properties,
            trusted_cross_agent=True,
        )

    async def _add_edge(
        self,
        source_id: str,
        target_id: str,
        label: str,
        properties: Optional[Dict],
        *,
        trusted_cross_agent: bool,
    ) -> None:
        """Implement ordinary and explicitly trusted edge writes atomically."""
        declared = properties.get("agent_id") if properties else None
        if declared is not None and not isinstance(declared, str):
            raise ValueError("Graph edge properties.agent_id must be a string")
        if self.agent_id and declared and declared != self.agent_id:
            raise ValueError("Graph edge owner does not match the bound agent")
        requested_owner = self.agent_id or declared or ""

        async with self.db.transaction():
            if self.db.backend_type == "postgres" and requested_owner:
                # Establish tenant scope before taking a physical/advisory row
                # lock.  Invalid ordinary edges must not let a tenant queue on
                # a foreign endpoint, while trusted lineage intentionally owns
                # only its source and never locks its foreign/absent target.
                preflight_owners = await self.db.fetchone(
                    "SELECT "
                    "EXISTS(SELECT 1 FROM graph_node_owners "
                    "       WHERE node_id = ? AND agent_id = ?), "
                    "EXISTS(SELECT 1 FROM graph_node_owners "
                    "       WHERE node_id = ? AND agent_id = ?), "
                    "EXISTS(SELECT 1 FROM graph_nodes WHERE node_id = ?)",
                    (
                        source_id,
                        requested_owner,
                        target_id,
                        requested_owner,
                        target_id,
                    ),
                )
                preflight_owns_source = bool(
                    preflight_owners and preflight_owners[0]
                )
                preflight_owns_target = bool(
                    preflight_owners and preflight_owners[1]
                )
                preflight_target_exists = bool(
                    preflight_owners and preflight_owners[2]
                )
                if trusted_cross_agent:
                    if not preflight_owns_source:
                        raise ValueError(
                            "Trusted cross-agent edge source is not owned by the bound agent"
                        )
                elif not (
                    preflight_owns_source
                    and preflight_owns_target
                    and preflight_target_exists
                ):
                    raise ValueError(
                        "Graph edge endpoints are not both owned by the bound agent"
                    )

            # Node deletion and ownership release take graph rows before any
            # edge ledger row. Ordinary edge admission must use the same order
            # and lock both endpoints together (sorted by the shared helper),
            # otherwise PostgreSQL can validate a soon-to-be-deleted witness
            # through MVCC and commit a dangling edge after node cleanup has
            # passed it. Trusted cross-agent edges deliberately permit an
            # absent/foreign target, so only their owned source participates.
            lock_ids = (
                [source_id]
                if self.db.backend_type == "postgres" and trusted_cross_agent
                else [source_id, target_id]
            )
            await lock_graph_nodes_for_update(
                self.db, lock_ids, agent_id=requested_owner
            )
            endpoint_rows = await self.db.fetchone(
                "SELECT "
                "EXISTS(SELECT 1 FROM graph_nodes WHERE node_id = ?), "
                "EXISTS(SELECT 1 FROM graph_nodes WHERE node_id = ?)",
                (source_id, target_id),
            )
            source_exists = bool(endpoint_rows and endpoint_rows[0])
            target_exists = bool(endpoint_rows and endpoint_rows[1])
            # Ordinary edges always require a materialized target. A bound
            # bootstrap writer may use a provisionally-reserved source: avatar
            # and backup workflows establish that source ownership witness
            # before the physical agent root exists. The ownership checks below
            # validate the reservation atomically. Unbound maintenance writers
            # have no such proof and must present a physical source too.
            if not requested_owner and (not source_exists or not target_exists):
                raise ValueError("Graph edge endpoints do not both exist")

            if requested_owner:
                endpoint_owners = await self.db.fetchone(
                    "SELECT "
                    "EXISTS(SELECT 1 FROM graph_node_owners "
                    "       WHERE node_id = ? AND agent_id = ?), "
                    "EXISTS(SELECT 1 FROM graph_node_owners "
                    "       WHERE node_id = ? AND agent_id = ?)",
                    (source_id, requested_owner, target_id, requested_owner),
                )
                owns_source = bool(endpoint_owners and endpoint_owners[0])
                owns_target = bool(endpoint_owners and endpoint_owners[1])
                if trusted_cross_agent:
                    if not owns_source:
                        raise ValueError(
                            "Trusted cross-agent edge source is not owned by the bound agent"
                        )
                elif not (owns_source and owns_target and target_exists):
                    raise ValueError(
                        "Graph edge endpoints are not both owned by the bound agent"
                    )

            existing = await self.db.fetchone(
                "SELECT properties FROM graph_edges "
                "WHERE source_id = ? AND target_id = ? AND label = ?",
                (source_id, target_id, label),
            )
            existing_owner_rows = await self.db.fetchall(
                "SELECT agent_id FROM graph_edge_owners "
                "WHERE source_id = ? AND target_id = ? AND label = ?",
                (source_id, target_id, label),
            )
            existing_owners = {row[0] for row in existing_owner_rows}
            if existing and requested_owner and not existing_owners:
                raise ValueError(
                    "Cannot claim or overwrite an unowned graph edge"
                )
            foreign_owned = bool(
                requested_owner
                and existing_owners
                and requested_owner not in existing_owners
            )
            if foreign_owned or (
                len(existing_owners) > 1
                and requested_owner
                and existing_owners != {requested_owner}
            ):
                raise ValueError(
                    "Cannot overwrite a graph edge owned by another agent"
                )

            current_owner_can_update = bool(
                not existing
                or not requested_owner
                or existing_owners == {requested_owner}
            )
            if current_owner_can_update:
                await self.db.execute(
                    self._upsert_edge_sql(),
                    (
                        source_id,
                        target_id,
                        label,
                        json.dumps(properties) if properties else None,
                    ),
                )

            if requested_owner:
                await record_graph_edge_owner(
                    self.db, source_id, target_id, label, requested_owner
                )
            else:
                # An unbound store may still connect two nodes with an
                # existing common owner (legacy/direct service usage).  Record
                # every common owner; different-owner endpoints yield none.
                if self.db.backend_type == "postgres":
                    owner_sql = (
                        "INSERT INTO graph_edge_owners "
                        "(source_id, target_id, label, agent_id) "
                        "SELECT ?, ?, ?, src.agent_id "
                        "FROM graph_node_owners src "
                        "JOIN graph_node_owners dst ON dst.agent_id = src.agent_id "
                        "WHERE src.node_id = ? AND dst.node_id = ? "
                        "ON CONFLICT DO NOTHING"
                    )
                else:
                    owner_sql = (
                        "INSERT OR IGNORE INTO graph_edge_owners "
                        "(source_id, target_id, label, agent_id) "
                        "SELECT ?, ?, ?, src.agent_id "
                        "FROM graph_node_owners src "
                        "JOIN graph_node_owners dst ON dst.agent_id = src.agent_id "
                        "WHERE src.node_id = ? AND dst.node_id = ?"
                    )
                await self.db.execute(
                    owner_sql,
                    (source_id, target_id, label, source_id, target_id),
                )

    async def delete_edge(self, source_id: str, target_id: str, label: str) -> None:
        """Release this tenant's edge witness, reclaiming only ownerless rows."""
        async with self.db.transaction():
            if self.agent_id:
                await self.db.execute(
                    "DELETE FROM graph_edge_owners "
                    "WHERE source_id = ? AND target_id = ? AND label = ? "
                    "AND agent_id = ?",
                    (source_id, target_id, label, self.agent_id),
                )
                await self.db.execute(
                    "DELETE FROM graph_edges "
                    "WHERE source_id = ? AND target_id = ? AND label = ? "
                    "AND NOT EXISTS ("
                    "  SELECT 1 FROM graph_edge_owners AS remaining_owner "
                    "  WHERE remaining_owner.source_id = graph_edges.source_id "
                    "  AND remaining_owner.target_id = graph_edges.target_id "
                    "  AND remaining_owner.label = graph_edges.label"
                    ")",
                    (source_id, target_id, label),
                )
                return

            await self.db.execute(
                "DELETE FROM graph_edge_owners "
                "WHERE source_id = ? AND target_id = ? AND label = ?",
                (source_id, target_id, label),
            )
            await self.db.execute(
                "DELETE FROM graph_edges "
                "WHERE source_id = ? AND target_id = ? AND label = ?",
                (source_id, target_id, label),
            )
    
    async def get_edges(self, node_id: str, direction: str = "both") -> List[Edge]:
        """Get edges connected to a node."""
        edges = []
        scope, scope_params = self._edge_scope()
        
        if direction in ("out", "both"):
            rows = await self.db.fetchall(
                "SELECT source_id, target_id, label, properties FROM graph_edges "
                f"WHERE source_id = ? AND {scope}",
                (node_id,) + scope_params,
            )
            edges.extend([
                Edge(
                    source_id=row[0],
                    target_id=row[1],
                    label=row[2],
                    properties=json.loads(row[3]) if row[3] else None
                )
                for row in rows
            ])
        
        if direction in ("in", "both"):
            rows = await self.db.fetchall(
                "SELECT source_id, target_id, label, properties FROM graph_edges "
                f"WHERE target_id = ? AND {scope}",
                (node_id,) + scope_params,
            )
            edges.extend([
                Edge(
                    source_id=row[0],
                    target_id=row[1],
                    label=row[2],
                    properties=json.loads(row[3]) if row[3] else None
                )
                for row in rows
            ])
        
        return edges
