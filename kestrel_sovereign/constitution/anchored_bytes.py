"""Read the constitution bytes an agent is anchored to.

One question, asked the same way by both reanchor entry points: *what is
actually stored under this agent's ``constitution_hash``, and if we cannot
show it, is that because nothing is there or because we cannot read it?*

The two answers are not interchangeable. **Absent** is the #2616 dangling
anchor a reanchor exists to repair — refusing it would brick the fix.
**Present but unreadable** could be hiding an active Amendment VIII, and an
irrevocable right whose precondition cannot be checked is not a right that may
be waived by accident (#2465).

The read is deliberately **unbound**. ``AsyncFileStore`` scopes an ordinary
read to ``file_owners``, and a row with no ownership entry comes back as
``None`` — indistinguishable from no row at all. That is not a corner case
here: ``file_owners`` arrived with #2649, every agent in the pre-#1118 cohort
this guard protects stored its constitution before that, and the backfill only
claims a blob when the agent carries a ``governed_by`` edge whose target equals
its ``constitution_hash`` — precisely the edge that has drifted in the #2616
population. Scoping this read would report "absent" for a constitution sitting
in the table byte-for-byte, and the guard would permit the erasure it exists to
prevent.

Reading it unbound is safe for the same reason it is necessary: the hash comes
from *this* agent's own node, and the store is content-addressed, so any row
under that hash holds exactly those bytes. Same argument as the unscoped
``governed_by`` read in ``constitution_reanchor._read_agent_anchor``.
"""

from __future__ import annotations

import logging
import hashlib
import json
import re
from copy import deepcopy
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)


async def store_verified_governing_file(
    storage, content: bytes, *, verification, artifact_content: bytes | None = None,
) -> str:
    """Native signed writers may restore this exact verified public blob owner.

    This is a first-party repair primitive inside an owning graph transaction,
    not a weaker generic file-store claim. The caller has verified its pinned
    external root and revalidates its governing witness before commit. Existing
    bytes must decrypt to the exact signed content; corruption is not replaced.
    """
    from kestrel_sovereign.constitution.amendment_artifact import AmendmentArtifactVerification
    digest = hashlib.sha256(content).hexdigest()
    if (
        not isinstance(verification, AmendmentArtifactVerification)
        or verification.ok is not True
        or verification.constitution_sha256 != digest
        or storage.owns_open_transaction is not True
        or not storage.files.agent_id
    ):
        raise RuntimeError("governing file restoration requires verified signed content and owned custody")
    if artifact_content is not None:
        artifact = json.loads(artifact_content)
        if (
            not isinstance(artifact, dict)
            or artifact.get("constitution_sha256") != digest
            or artifact.get("signer") != verification.signer
        ):
            raise RuntimeError("signed artifact does not describe the verified governing content")
    await _store_exact_governing_file(storage, content, "KESTREL_CONSTITUTION.md")
    if artifact_content is not None:
        # The callers supply the exact detached artifact they verified, not
        # its serialization rebuilt from a mapping. Both public witnesses use
        # the same physical-byte and tenant-custody primitive.
        await _store_exact_governing_file(
            storage, artifact_content, "KESTREL_CONSTITUTION.reanchor.signed.json",
        )
    return digest


async def _store_exact_governing_file(storage, content: bytes, name: str) -> None:
    """Publish native public bytes after signed or single-use bootstrap authority.

    Private first-party primitive: callers must establish their authority and
    complete graph reservation first. Privacy-facing file caches are neither
    publication nor an attestation of the actual native conflict winner.
    """
    files = storage.files
    if storage.owns_open_transaction is not True or not files.agent_id:
        raise RuntimeError("governing publication requires owned native tenant custody")
    await _store_exact_native_file(storage.db, files, content, name)


async def _store_exact_native_file(db, files, content: bytes, name: str, *, metadata=None) -> str:
    """First-party publication after complete graph reservations and authority.

    Birth copies use an owned, digest-verified source; inception uses resolved
    new-identity bytes. Signed repair establishes its external authority before
    calling this same native conflict-winner/tenant-owner attestation.
    """
    from kestrel_sovereign.storage.async_file_store import AsyncFileStore

    if db.owns_open_transaction is not True or not files.agent_id:
        raise RuntimeError("exact file publication requires owned native tenant custody")
    digest = hashlib.sha256(content).hexdigest()
    unbound = AsyncFileStore(db)
    lock = " FOR UPDATE" if db.backend_type == "postgres" else ""
    row = await db.fetchone(
        "SELECT content_hash FROM files WHERE content_hash = ?" + lock, (digest,),
    )
    if row is None:
        # Canonical native storage owns hashing, encryption and size limits.
        # Another creator may win the absent-row race; INSERT ignores that
        # conflict, so lock and validate the ACTUAL winner below, not our input.
        await unbound.store_file(content, name)
        row = await db.fetchone(
            "SELECT content_hash FROM files WHERE content_hash = ?" + lock, (digest,),
        )
        if row is None:
            raise RuntimeError("file disappeared before physical custody acquisition")
    existing = await unbound.retrieve_file(digest)
    if existing != content:
        raise RuntimeError("stored file bytes do not verify against exact publication content")
    # Restore only a missing ownership witness. Existing per-tenant name and
    # provenance are not replaced. Shared blob metadata may belong to another
    # tenant; verified public bytes confer no authority over that provenance.
    await db.execute(
        "INSERT OR IGNORE INTO file_owners (content_hash, agent_id, original_name, metadata) VALUES (?, ?, ?, ?)",
        (digest, files.agent_id, name, json.dumps(metadata) if metadata else None),
    )
    if db.backend_type == "postgres":
        # Conflict-ignore insertion does not retain an existing owner's row.
        # Hold that exact tenant witness after the shared blob (the same order
        # as native exit), or refuse if it disappeared before custody arrived.
        owner = await db.fetchone(
            "SELECT content_hash FROM file_owners WHERE content_hash = ? AND agent_id = ? FOR UPDATE",
            (digest, files.agent_id),
        )
        if owner is None:
            raise RuntimeError("governing file ownership disappeared during publication")
    return digest


async def lock_governing_file(storage, digest: str) -> bytes:
    """After graph custody, retain and verify the actual tenant-owned blob."""
    if storage.owns_open_transaction is not True or not storage.files.agent_id:
        raise RuntimeError("governing attestation requires owned native tenant custody")
    lock = " FOR UPDATE" if storage.db.backend_type == "postgres" else ""
    blob = await storage.db.fetchone(
        "SELECT content_hash FROM files WHERE content_hash = ?" + lock, (digest,),
    )
    owner = await storage.db.fetchone(
        "SELECT content_hash FROM file_owners WHERE content_hash = ? AND agent_id = ?" + lock,
        (digest, storage.files.agent_id),
    )
    if blob is None or owner is None:
        raise RuntimeError("Anchored constitution blob is missing or its ownership custody is absent")
    # Always the bound native reader: no session cache can attest to bytes in
    # the locked database row. Decryption failure refuses the owning commit.
    content = await storage.files.retrieve_file(digest)
    if content is None or hashlib.sha256(content).hexdigest() != digest:
        raise RuntimeError("Anchored constitution physical bytes fail content-address verification")
    return content


def historical_anchor_hash(
    properties: Mapping, governed_by_targets: Iterable[str],
) -> Optional[str]:
    """Recover evidence, never authority, when the operative pointer is lost.

    Native current/history receipts survive edge/pointer deletion. Inspect
    their typed hash fields, not arbitrary receipt prose. Without the pointer
    conflicting, malformed or excessive evidence must fail closed; an
    operator can restore the exact prior pointer before attempting repair.
    An intact pointer remains authoritative: old receipt history normally
    names multiple superseded constitutions and is not a competing pointer.
    """
    pointer = properties.get("constitution_hash")
    from kestrel_sovereign.constitution.reanchor_receipt import validate_constitution_reanchor_evidence

    validate_constitution_reanchor_evidence(properties)
    if pointer:
        return pointer
    candidates: set[str] = set()

    def add(value, *, absent_ok=True):
        if value is None and absent_ok:
            return
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("Missing anchor pointer has unreadable historical governance receipt evidence")
        candidates.add(value)

    targets = tuple(governed_by_targets)
    if len(targets) > 128:
        raise ValueError("Missing anchor pointer has excessive historical governance evidence")
    for target in targets:
        add(target, absent_ok=False)

    def inspect(receipt, kind):
        if not isinstance(receipt, Mapping):
            raise ValueError("Missing anchor pointer has unreadable historical governance receipt evidence")
        # A reanchor always has a destination. Its optional prior hash cannot
        # substitute for a missing/null destination and authorize superseded
        # bytes as the only surviving history. Genesis likewise needs a hash.
        required = "constitution_hash" if kind == "genesis_audit" else "new_hash"
        add(receipt.get(required), absent_ok=False)
        if kind == "constitution_reanchor":
            old_hash = receipt.get("old_hash")
            if old_hash != "none":
                add(old_hash)

    for kind in ("genesis_audit", "constitution_reanchor"):
        current = properties.get(kind)
        if current is not None:
            inspect(current, kind)
        history = properties.get(kind + "_history")
        if history is None:
            continue
        if not isinstance(history, list) or len(history) > 128:
            raise ValueError("Missing anchor pointer has unreadable or excessive historical governance receipt evidence")
        for entry in history:
            if not isinstance(entry, Mapping):
                raise ValueError("Missing anchor pointer has unreadable historical governance receipt evidence")
            inspect(entry.get("receipt"), kind)
            add(entry.get("superseded_by_constitution_hash"), absent_ok=False)
    if len(candidates) > 1:
        raise ValueError("Missing anchor pointer has ambiguous historical governance; restore its exact prior pointer before signed repair")
    return next(iter(candidates), None)


def governance_evidence(properties: Mapping, governed_by_targets: Iterable[str]) -> dict:
    """Snapshot the exact governing facts validated by signed repair preflight.

    Non-governance metadata is excluded so unrelated updates are preserved.
    Copies are essential: a caller mutating its graph node must not mutate the
    comparison witness along with it. This evidence is not signing authority.
    """
    return {
        "properties": deepcopy({key: properties.get(key) for key in (
            "constitution_hash", "emancipation_contract", "genesis_audit",
            "genesis_audit_history", "constitution_reanchor", "constitution_reanchor_history",
        )}),
        "governed_by_targets": sorted(set(governed_by_targets)),
    }


async def lock_governance_rows(
    storage, agent_id: str, *, required_target: str | None = None,
    expected_targets: Iterable[str] | None = None,
) -> None:
    """After graph reservation, retain the physical governing row witness.

    All publishers use ownership-before-edge order, matching native deletion.
    SQLite's owning writer supplies serialization, not existence. Both
    backends must read the actual required witnesses.
    """
    postgres = storage.db.backend_type == "postgres"
    lock = " FOR UPDATE" if postgres else ""
    identity = await storage.db.fetchone(
        "SELECT node_id FROM graph_nodes WHERE node_id = ?" + lock, (agent_id,),
    )
    identity_owners = await storage.db.fetchall(
        "SELECT node_id FROM graph_node_owners WHERE node_id = ? AND agent_id = ?" + lock,
        (agent_id, agent_id),
    )
    if identity is None or not identity_owners:
        raise RuntimeError("Agent identity or its ownership disappeared before governing custody")
    if required_target is not None:
        target = await storage.db.fetchone(
            "SELECT node_id FROM graph_nodes WHERE node_id = ?" + lock,
            (required_target,),
        )
        target_owner = await storage.db.fetchone(
            "SELECT node_id FROM graph_node_owners WHERE node_id = ? AND agent_id = ?" + lock,
            (required_target, agent_id),
        )
        if target is None or target_owner is None:
            raise RuntimeError("Governing target node or its ownership custody is absent")
    # Never lock an endpoint outside the graph reservation set. Exit needs
    # only its current governing edge; signed repair/genesis pass their
    # complete captured set and reject changes without extending custody.
    targets = sorted(set(expected_targets)) if expected_targets is not None else (
        [required_target] if required_target is not None else []
    )
    edge_owners, edges = [], []
    if targets:
        predicate = " AND target_id IN (" + ",".join("?" for _ in targets) + ")"
        params = (agent_id, *targets)
        canonical_target = 'target_id COLLATE "C"' if postgres else "target_id"
        canonical_owner = 'agent_id COLLATE "C"' if postgres else "agent_id"
        edge_owners = await storage.db.fetchall(
            "SELECT target_id, agent_id FROM graph_edge_owners WHERE source_id = ? AND label = 'governed_by'"
            + predicate + f" ORDER BY {canonical_target}, {canonical_owner}" + lock, params,
        )
        edges = await storage.db.fetchall(
            "SELECT target_id FROM graph_edges WHERE source_id = ? AND label = 'governed_by'"
            + predicate + f" ORDER BY {canonical_target}" + lock, params,
        )
    if expected_targets is not None and {row[0] for row in edges} != set(targets):
        raise RuntimeError("Captured governing edges disappeared before physical custody")
    if required_target is not None and (
        (required_target, agent_id) not in edge_owners
        or (required_target,) not in edges
    ):
        raise RuntimeError("Missing or mis-targeted governed_by edge or ownership custody")


async def revalidate_governance_evidence(
    storage, agent_id: str, expected: dict, *, required_target: str | None = None,
):
    """Under the writer's graph locks, refuse a changed preflight witness.

    Both native signed writers use this before any governance mutation, inside
    their owning transaction. A changed pointer/receipt/rights/edge set requires
    a fresh inspection and authorization, never adoption of a newer CAS fence.
    """
    await lock_governance_rows(
        storage, agent_id, expected_targets=expected["governed_by_targets"],
        required_target=required_target,
    )
    node = await storage.get_node(agent_id)
    if node is None or node.node_type != "agent":
        raise RuntimeError("Agent identity disappeared during signed repair")
    rows = await storage.db.fetchall(
        "SELECT target_id FROM graph_edges WHERE source_id = ? AND label = 'governed_by'",
        (agent_id,),
    )
    if governance_evidence(node.properties, (row[0] for row in rows)) != expected:
        raise RuntimeError("Signed repair governing evidence changed; reload and reauthorize before repair")
    return node


async def read_anchored_constitution(
    db: "AsyncDatabase", anchored_hash: str
) -> Tuple[Optional[str], bool]:
    """Return ``(text, present)`` for the blob stored under ``anchored_hash``.

    ``(None, False)`` — nothing is stored under that hash. ABSENT.
    ``(None, True)``  — something is, and this process cannot turn it into
    text: a wrong ``KESTREL_DATA_KEY``, corruption, or bytes that are not
    UTF-8. UNREADABLE.
    ``(text, True)``  — the anchored constitution.
    """
    from kestrel_sovereign.security.encryption import DecryptionError
    from kestrel_sovereign.storage.async_file_store import AsyncFileStore

    # No agent_id: see the module docstring. This is the whole point.
    store = AsyncFileStore(db)
    try:
        raw = await store.retrieve_file(anchored_hash)
    except DecryptionError:
        # UNREADABLE means "the bytes are there and this process cannot open
        # them" — a wrong KESTREL_DATA_KEY. Deliberately narrow: a dropped
        # connection is not a key problem, and swallowing it here would tell
        # the operator to go check their data key. Every caller already has a
        # boundary that names a database failure for what it is, so those
        # propagate to it.
        logger.warning(
            "The constitution stored under %s could not be decrypted",
            anchored_hash[:12],
            exc_info=True,
        )
        return None, True
    if raw is None:
        return None, False
    if hashlib.sha256(raw).hexdigest() != anchored_hash:
        logger.warning("Stored historical constitution fails its addressed hash: %s", anchored_hash[:12])
        return None, True
    try:
        return raw.decode("utf-8"), True
    except UnicodeDecodeError:
        logger.warning(
            "The constitution stored under %s is not UTF-8 text",
            anchored_hash[:12],
            exc_info=True,
        )
        return None, True
