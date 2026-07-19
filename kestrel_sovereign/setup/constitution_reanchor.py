"""Constitution reanchor — write path for ``kestrel constitution reanchor``.

Drift detection ships in :mod:`kestrel_sovereign.doctor`. This module is
the writer side: when an agent's anchored ``constitution_hash`` no longer
matches the canonical file on disk, this helper updates **all five**
governance locations in a single AsyncStorage transaction, stores the verified
authorization artifact, and takes a timestamped file-level backup so a botched
run is recoverable.

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
move to the new hash. The edge replacement prunes EVERY ``governed_by``
edge whose target is not the new hash — not just the edge named by the
agent's ``constitution_hash`` property (#2617): in the exact drift state
that motivates a reanchor, property and edge disagree, so a
property-derived delete would miss the actual stale edge and leave the
agent with two governing constitutions.

When the anchor is already current but stray non-target ``governed_by``
edges exist (the #2617 aftermath state), a forced run with a verified
signed artifact performs a prune-only write (backup + transaction) so
already-drifted DBs can converge without changing the anchor. A dry run
reports those stale edges as drift.

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

from kestrel_sovereign.constitution.emancipation import (
    EmancipationConfigError,
    EmancipationContract,
    check_iron_rule,
    contract_from_json,
    contract_to_json,
    parse_emancipation_block,
)
from kestrel_sovereign.constitution.amendment_artifact import (
    AmendmentArtifactError,
    AmendmentArtifactVerification,
    load_verified_reanchor_artifact,
)
from kestrel_sovereign.constitution.resolver import (
    resolve_governing_constitution_bytes,
)
from kestrel_sovereign.constitution.trust_root import (
    SovereignTrustRootError,
    load_sovereign_trust_root,
)
from kestrel_sovereign.storage import AsyncStorage, GraphNode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReanchorResult:
    """Outcome of :func:`reanchor_constitution`.

    Exactly one of ``unchanged`` / ``drift_unforced`` / ``reanchored`` /
    ``iron_rule_violation`` (set as ``error``) is True. The CLI dispatches
    messaging on this.
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
    #: When set, ``error`` is a #1118 Iron Rule refusal (not a generic
    #: failure). The CLI uses this to print the diff-clause rather than
    #: a stack trace.
    iron_rule_violation: str | None = None
    #: ``governed_by`` edge targets that are neither the anchored property
    #: hash nor the canonical hash — dangling governance edges (#2617).
    #: Populated on the read side so a dry run can report them as drift.
    stale_edges: tuple[str, ...] = ()
    #: Non-target ``governed_by`` edge targets actually deleted by a write.
    #: On ``reanchored`` results this excludes the expected ``old_hash``
    #: edge; on ``unchanged`` results a non-empty value means the run was
    #: a prune-only cleanup of stale edges.
    pruned_stale_edges: tuple[str, ...] = ()


async def reanchor_constitution(
    *,
    agent_name: str,
    agent_dir: Path,
    canonical_path: Path,
    force: bool,
    authorization: str = "kestrel constitution reanchor",
    kestrel_toml_path: Path | None = None,
    amendment_artifact_path: Path | None = None,
    sovereign_trust_root_path: Path | None = None,
) -> ReanchorResult:
    """Reanchor one agent to the current canonical constitution.

    Read order matters: we open the DB twice. First (read-only) to
    discover whether drift exists; if no drift OR no ``--force``, we
    return without ever taking the destructive backup. Only when we're
    actually going to write do we copy the DB aside and reopen the
    storage layer for mutation.

    Drift includes dangling governance edges (#2617): when the anchored
    hash already matches the canonical hash but stray non-target
    ``governed_by`` edges exist, an unforced run reports them via
    ``drift_unforced`` + ``stale_edges``, and a forced run with a
    verified signed artifact performs a prune-only write (``unchanged``
    + ``pruned_stale_edges``) without touching the anchor itself.

    Amendment VIII handling (#1118):

      1. The agent's anchored ``EmancipationContract`` (if any) is
         loaded from ``agent.properties.emancipation_contract``.
      2. If a ``kestrel_toml_path`` is provided, its ``[emancipation]``
         block is parsed and compared to the anchored contract via
         :func:`check_iron_rule`. Any narrowing transition refuses the
         reanchor — no backup is taken, no write happens.
      3. The effective contract (anchored, if active; otherwise the
         candidate from the new block, if active; otherwise None) is
         applied to the canonical markdown via :func:`apply_emancipation`
         **before** the new hash is computed. This is what makes the
         active form survive reanchor — without this step the canonical
         dormant text would silently overwrite the Sovereign-authored
         contract.

    Args:
        agent_name: Display name (used for messages and backup naming).
        agent_dir: Agent's data directory (contains ``kestrel_prime.db``).
        canonical_path: On-disk constitution to anchor against.
        force: Required for any write. Without it, drift is reported
            but the DB is not touched.
        authorization: Free-form string stored in the audit record so
            future readers know who performed this reanchor.
        kestrel_toml_path: Optional path to the project's
            ``kestrel.toml``. When provided, the ``[emancipation]``
            block is parsed and Iron-Rule-checked against the agent's
            anchored contract. When None, no comparison is made and the
            anchored contract is preserved as-is.
        amendment_artifact_path: Detached Sovereign-signed reanchor artifact.
            Required before any forced write, including an emancipation
            sidecar backfill.
        sovereign_trust_root_path: Optional explicit operator-owned JSON DID
            document. The shared resolver also reads
            ``KESTREL_SOVEREIGN_TRUST_ROOT_PATH`` and rejects conflicts.
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

    # Pre-flight the canonical source so an unreadable path returns a clean
    # ReanchorResult error rather than blowing up inside the resolver below.
    try:
        canonical_path.read_bytes()
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

    # REFUSE non-authoritative sources (#2463 review): the periodic integrity
    # audit recomputes from the packaged governing source, so reanchoring to any
    # other file would produce an agent guaranteed to fail its next audit. A
    # legitimate custom governing source is expressed by pointing
    # config.CONSTITUTION_PATH at it, not by passing an arbitrary --constitution-path.
    from kestrel_sovereign.constitution.resolver import (
        is_authoritative_governing_source,
    )

    if not is_authoritative_governing_source(str(canonical_path)):
        return ReanchorResult(
            agent_name=agent_name,
            db_path=db_path,
            canonical_path=canonical_path,
            old_hash=None,
            new_hash=None,
            backup_path=None,
            error=(
                f"Refusing to reanchor to non-authoritative constitution source "
                f"{canonical_path}: the periodic integrity audit recomputes from "
                f"the packaged governing source, so an agent anchored elsewhere "
                f"would fail its next audit and Safe-Mode. Reanchor against the "
                f"packaged source (omit --constitution-path) or point "
                f"config.CONSTITUTION_PATH at your authoritative source (#2463)."
            ),
        )

    (
        old_hash,
        agent_did,
        anchored_contract_json,
        governed_edge_targets,
    ) = await _read_agent_anchor(db_path)
    if old_hash is None:
        return ReanchorResult(
            agent_name=agent_name,
            db_path=db_path,
            canonical_path=canonical_path,
            old_hash=None,
            new_hash=None,
            backup_path=None,
            error=(
                "Agent has no constitution_hash property. "
                "Re-incept the agent rather than reanchoring."
            ),
        )

    # --- #1118: Iron Rule + active-form re-application -------------------
    try:
        anchored_contract = contract_from_json(anchored_contract_json)
    except EmancipationConfigError as exc:
        return ReanchorResult(
            agent_name=agent_name, db_path=db_path, canonical_path=canonical_path,
            old_hash=old_hash, new_hash=None, backup_path=None,
            error=(
                f"Anchored emancipation_contract is corrupted: {exc}. "
                f"Refusing to reanchor without a clean structured receipt."
            ),
        )

    candidate_contract: EmancipationContract | None = None
    if kestrel_toml_path is not None and kestrel_toml_path.exists():
        try:
            from kestrel_sovereign.setup.toml_file import read_toml
            candidate_contract = parse_emancipation_block(read_toml(kestrel_toml_path))
        except EmancipationConfigError as exc:
            return ReanchorResult(
                agent_name=agent_name, db_path=db_path, canonical_path=canonical_path,
                old_hash=old_hash, new_hash=None, backup_path=None,
                error=(
                    f"[emancipation] block in {kestrel_toml_path} is invalid: {exc}. "
                    f"Refusing reanchor — fix the block or remove it."
                ),
            )

    violation = check_iron_rule(
        anchored=anchored_contract,
        candidate=candidate_contract,
    )
    if violation is not None:
        # Refuse cleanly; no backup, no write, no DB touch beyond the
        # earlier read-only inspection.
        return ReanchorResult(
            agent_name=agent_name, db_path=db_path, canonical_path=canonical_path,
            old_hash=old_hash, new_hash=None, backup_path=None,
            error=violation,
            iron_rule_violation=violation,
        )

    # Pick the contract to anchor: anchored takes precedence (frozen
    # post-activation per #1118); otherwise activate the candidate if it
    # asks for activation.
    effective_contract = anchored_contract if (
        anchored_contract is not None and anchored_contract.enabled
    ) else candidate_contract

    # Route through the single production resolver (#2463) so reanchor produces
    # byte-identical governing content to inception + verification, pointed at
    # the same ``canonical_path``.
    new_content = resolve_governing_constitution_bytes(
        effective_contract if (
            effective_contract is not None and effective_contract.enabled
        ) else None,
        constitution_path=str(canonical_path),
    )

    new_hash = hashlib.sha256(new_content).hexdigest()

    # #1118 sidecar backfill: if the agent has active-form bytes anchored
    # (e.g. it was incepted between #1112 — which added activation at
    # inception — and #1118 — which added the JSON sidecar), the
    # constitution hash will already match the active form but
    # ``agent.properties.emancipation_contract`` will be missing. Without
    # a backfill, a future reanchor with no [emancipation] block would
    # treat the agent as having no anchored contract and could overwrite
    # the active form with canonical dormant text. Force the write path
    # in that specific case so the receipt lands.
    needs_sidecar_backfill = (
        anchored_contract is None
        and candidate_contract is not None
        and candidate_contract.enabled
    )

    # Dangling governance edges (#2617): ``governed_by`` targets that are
    # neither the anchored property hash nor the canonical hash. In the
    # drift state that motivates a reanchor, property and edge disagree, so
    # these are exactly the edges a property-derived delete would strand.
    stale_edges = tuple(
        target
        for target in governed_edge_targets
        if target not in (old_hash, new_hash)
    )

    if old_hash == new_hash and not needs_sidecar_backfill:
        if not stale_edges:
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
            # Anchor is current but stray governance edges exist — report
            # them as drift so a dry run surfaces the inconsistency.
            return ReanchorResult(
                agent_name=agent_name,
                db_path=db_path,
                canonical_path=canonical_path,
                old_hash=old_hash,
                new_hash=new_hash,
                backup_path=None,
                drift_unforced=True,
                stale_edges=stale_edges,
            )
        # Prune-only write path (#2617 one-shot cleanup): the anchor does
        # not change, but converging the governance edges is still a
        # governance write — it runs under the SAME signed-artifact +
        # trust-root gate as a full reanchor, with the same backup.
        if amendment_artifact_path is None:
            return ReanchorResult(
                agent_name=agent_name,
                db_path=db_path,
                canonical_path=canonical_path,
                old_hash=old_hash,
                new_hash=new_hash,
                backup_path=None,
                stale_edges=stale_edges,
                error=(
                    "A Sovereign-signed amendment artifact is required to "
                    "prune stale governed_by edges (governance write). Pass "
                    "--signed-artifact and configure the external Sovereign "
                    "trust root."
                ),
            )
        try:
            _load_verified_authorization(
                amendment_artifact_path=amendment_artifact_path,
                sovereign_trust_root_path=sovereign_trust_root_path,
                agent_did=agent_did,
                new_hash=new_hash,
            )
        except (SovereignTrustRootError, AmendmentArtifactError) as exc:
            return ReanchorResult(
                agent_name=agent_name,
                db_path=db_path,
                canonical_path=canonical_path,
                old_hash=old_hash,
                new_hash=new_hash,
                backup_path=None,
                stale_edges=stale_edges,
                error=str(exc),
            )

        backup_path = _backup_db(db_path)
        logger.info(
            "Backed up agent DB to %s before stale-edge prune", backup_path
        )
        try:
            pruned = await _prune_stale_governance_edges(
                db_path=db_path, agent_did=agent_did, keep_hash=new_hash
            )
        except Exception as exc:  # noqa: BLE001 — surface verbatim
            logger.exception(
                "Stale-edge prune failed; backup retained at %s", backup_path
            )
            return ReanchorResult(
                agent_name=agent_name,
                db_path=db_path,
                canonical_path=canonical_path,
                old_hash=old_hash,
                new_hash=new_hash,
                backup_path=backup_path,
                stale_edges=stale_edges,
                error=(
                    f"Stale-edge prune failed mid-write ({exc!r}). "
                    f"DB backup at {backup_path}"
                ),
            )
        return ReanchorResult(
            agent_name=agent_name,
            db_path=db_path,
            canonical_path=canonical_path,
            old_hash=old_hash,
            new_hash=new_hash,
            backup_path=backup_path,
            unchanged=True,
            stale_edges=stale_edges,
            pruned_stale_edges=tuple(pruned),
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
            stale_edges=stale_edges,
        )

    # Authorization is a pre-write gate. The graph DB is the object being
    # protected, so neither its root properties nor any material derived from
    # them is consulted here (#2499). Complete root resolution, artifact IO,
    # and signature verification before even taking the backup.
    if amendment_artifact_path is None:
        return ReanchorResult(
            agent_name=agent_name,
            db_path=db_path,
            canonical_path=canonical_path,
            old_hash=old_hash,
            new_hash=new_hash,
            backup_path=None,
            stale_edges=stale_edges,
            error=(
                "A Sovereign-signed amendment artifact is required for a "
                "forced reanchor. Pass --signed-artifact and configure the "
                "external Sovereign trust root."
            ),
        )

    try:
        (
            amendment_artifact_bytes,
            amendment_artifact,
            amendment_verification,
        ) = _load_verified_authorization(
            amendment_artifact_path=amendment_artifact_path,
            sovereign_trust_root_path=sovereign_trust_root_path,
            agent_did=agent_did,
            new_hash=new_hash,
        )
    except (SovereignTrustRootError, AmendmentArtifactError) as exc:
        return ReanchorResult(
            agent_name=agent_name,
            db_path=db_path,
            canonical_path=canonical_path,
            old_hash=old_hash,
            new_hash=new_hash,
            backup_path=None,
            stale_edges=stale_edges,
            error=str(exc),
        )

    backup_path = _backup_db(db_path)
    logger.info("Backed up agent DB to %s before reanchor", backup_path)

    # If we're activating dormant→active at reanchor (anchored had no
    # contract, candidate is active), the JSON receipt needs to be
    # written too. If anchored was already active, we re-write the same
    # receipt (cheap idempotent upsert keeps the property consistent).
    contract_json_to_write = (
        contract_to_json(effective_contract)
        if effective_contract is not None and effective_contract.enabled
        else None
    )

    try:
        pruned = await _write_reanchor(
            db_path=db_path,
            agent_did=agent_did,
            old_hash=old_hash,
            new_hash=new_hash,
            new_content=new_content,
            canonical_path=canonical_path,
            authorization=authorization,
            emancipation_contract_json=contract_json_to_write,
            amendment_artifact_path=amendment_artifact_path,
            amendment_artifact_bytes=amendment_artifact_bytes,
            amendment_artifact=amendment_artifact,
            amendment_verification=amendment_verification,
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
            stale_edges=stale_edges,
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
        stale_edges=stale_edges,
        # The old_hash edge is the expected replacement; anything else
        # pruned was a dangling governance edge (#2617).
        pruned_stale_edges=tuple(t for t in pruned if t != old_hash),
    )


async def _read_agent_anchor(
    db_path: Path,
) -> tuple[str | None, str, dict | None, tuple[str, ...]]:
    """Return ``(constitution_hash, agent_did, emancipation_contract_json,
    governed_by_edge_targets)``.

    Read-only — safe to call before deciding whether to touch the DB.
    Returns ``(None, "", None, ())`` if the agent node has no anchored hash.
    The contract field is ``None`` for dormant agents and for legacy
    agents incepted before #1118 (no JSON receipt was written). The edge
    targets are ALL current ``governed_by`` targets, so callers can detect
    dangling governance edges the property alone would hide (#2617).
    """
    async with AsyncStorage(str(db_path)) as storage:
        agent_nodes = await storage.graph.get_nodes_by_type("agent")
        if not agent_nodes:
            return None, "", None, ()
        agent = agent_nodes[0]
        edges = await storage.graph.get_edges(agent.node_id, direction="out")
        governed_targets = tuple(
            edge.target_id
            for edge in edges or []
            if edge.label == "governed_by"
        )
        return (
            agent.properties.get("constitution_hash"),
            agent.node_id,
            agent.properties.get("emancipation_contract"),
            governed_targets,
        )


def _load_verified_authorization(
    *,
    amendment_artifact_path: Path,
    sovereign_trust_root_path: Path | None,
    agent_did: str,
    new_hash: str,
) -> tuple[bytes, dict, AmendmentArtifactVerification]:
    """Resolve the operator trust root and verify the signed artifact.

    Shared pre-write gate for both the full reanchor and the prune-only
    stale-edge cleanup (#2617). Raises ``SovereignTrustRootError`` /
    ``AmendmentArtifactError`` on any failure — callers convert those to
    a ``ReanchorResult`` error before any backup or write happens.
    """
    trusted_did_document = load_sovereign_trust_root(
        explicit_path=sovereign_trust_root_path,
        agent_dids={agent_did},
    )
    return load_verified_reanchor_artifact(
        amendment_artifact_path,
        trusted_did_document=trusted_did_document,
        expected_constitution_sha256=new_hash,
    )


async def _delete_nontarget_governance_edges(
    storage: AsyncStorage, agent_did: str, keep_hash: str
) -> list[str]:
    """Delete every ``governed_by`` edge from ``agent_did`` whose target is
    not ``keep_hash``; return the deleted targets.

    This is the #2617 fix: deleting only the property-derived old hash
    misses the actual stale edge whenever property and edge disagree —
    the exact drift state that motivates a reanchor. Callers must invoke
    this inside their transaction, AFTER adding the ``keep_hash`` edge,
    so a concurrent reader never sees zero governing constitutions.
    """
    edges = await storage.graph.get_edges(agent_did, direction="out")
    pruned: list[str] = []
    for edge in edges or []:
        if edge.label == "governed_by" and edge.target_id != keep_hash:
            await storage.graph.delete_edge(
                agent_did, edge.target_id, "governed_by"
            )
            pruned.append(edge.target_id)
    return pruned


async def _prune_stale_governance_edges(
    *, db_path: Path, agent_did: str, keep_hash: str
) -> list[str]:
    """Prune-only write: converge ``governed_by`` edges on ``keep_hash``.

    Used when the anchor is already current but dangling edges remain
    (#2617 one-shot cleanup). The anchored edge is re-asserted first
    (idempotent upsert) so the prune can never leave the agent without a
    governing edge, then every non-target edge is deleted — all inside
    one transaction with automatic rollback on failure.
    """
    async with AsyncStorage(str(db_path)) as storage:
        async with storage.db.transaction():
            await storage.graph.add_edge(agent_did, keep_hash, "governed_by")
            return await _delete_nontarget_governance_edges(
                storage, agent_did, keep_hash
            )


async def _write_reanchor(
    *,
    db_path: Path,
    agent_did: str,
    old_hash: str,
    new_hash: str,
    new_content: bytes,
    canonical_path: Path,
    authorization: str,
    emancipation_contract_json: dict | None,
    amendment_artifact_path: Path,
    amendment_artifact_bytes: bytes,
    amendment_artifact: dict,
    amendment_verification: AmendmentArtifactVerification,
) -> list[str]:
    """Apply the five governance locations plus authorization atomically.

    Returns the ``governed_by`` edge targets deleted in step 3 (the
    expected ``old_hash`` edge plus any dangling edges, #2617).

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
      3. Replace the governed_by edges: add new first, then delete EVERY
         governed_by edge whose target is not the new hash (#2617 — not
         just the property-derived old hash, which misses the actual
         stale edge when property and edge disagree). Add-first ordering
         means a concurrent reader inside the transaction (if any) never
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

            artifact_hash = await storage.files.store_file(
                amendment_artifact_bytes,
                "KESTREL_CONSTITUTION.reanchor.signed.json",
            )
            expected_artifact_hash = hashlib.sha256(
                amendment_artifact_bytes
            ).hexdigest()
            if artifact_hash != expected_artifact_hash:
                raise RuntimeError(
                    "Artifact store hash mismatch: stored "
                    f"{artifact_hash}, expected {expected_artifact_hash}"
                )

            # 2. Document graph node for the new constitution. Create only
            # when missing: add_node is a full-properties upsert, and when
            # re-anchoring BACK to a previously anchored hash the existing
            # node's inception metadata (created_at) must survive.
            if await storage.graph.get_node(new_hash) is None:
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
            await storage.graph.add_node(
                GraphNode(
                    node_id=artifact_hash,
                    node_type="constitution_amendment_artifact",
                    label="Signed Constitution Reanchor Artifact",
                    properties={
                        "hash": artifact_hash,
                        "type": "SignedConstitutionAmendment",
                        "artifact_type": amendment_artifact.get("artifact_type"),
                        "constitution_hash": new_hash,
                        "signer": amendment_verification.signer,
                        "source_path": str(amendment_artifact_path),
                        "created_at": amendment_artifact.get("created_at"),
                        "anchored_at": _now_iso(),
                        "verification": amendment_verification.reason,
                    },
                )
            )

            # 3. Replace the governed_by edges — add new first, then prune
            # every non-target edge (#2617).
            await storage.graph.add_edge(agent_did, new_hash, "governed_by")
            pruned = await _delete_nontarget_governance_edges(
                storage, agent_did, new_hash
            )

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
            from kestrel_sovereign.constitution.genesis_audit import (
                supersede_genesis_audit,
            )

            supersede_genesis_audit(
                agent.properties,
                constitution_hash=new_hash,
                provenance="setup:constitution_reanchor",
            )
            agent.properties["constitution_reanchor"] = {
                "timestamp": _now_iso(),
                "old_hash": old_hash,
                "new_hash": new_hash,
                "source_path": str(canonical_path),
                "authorization": authorization,
                "signed_artifact_hash": artifact_hash,
                "signed_artifact_path": str(amendment_artifact_path),
                "signed_artifact_signer": amendment_verification.signer,
                "signed_artifact_verification": amendment_verification.reason,
            }
            # Anchor (or refresh) the structured contract receipt.
            # Idempotent for the unchanged-active case; performs the
            # dormant→active activation when reanchor enables Amendment
            # VIII for the first time.
            if emancipation_contract_json is not None:
                agent.properties["emancipation_contract"] = emancipation_contract_json
            await storage.graph.add_node(agent)
    return pruned


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
