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

We also record an audit entry at ``agent.properties.constitution_reanchor``
(timestamp + old_hash + new_hash + source_path), and append the receipt it
replaces to ``agent.properties.constitution_reanchor_history`` — see
:mod:`kestrel_sovereign.constitution.reanchor_receipt`. Until #2963 this
docstring said "append" while the code assigned, so each reanchor destroyed the
previous receipt; the runtime ``!reanchor-constitution`` chat command shares the
same helper but still records ``path`` where this writer records ``source_path``.

We do NOT delete the old document node or the old file blob — they're
retained for audit. Only the ``governed_by`` edge and the RAG chunks
move to the new hash.

Drift is NOT just a hash mismatch. The integrity audit's proof 2 requires
the ``governed_by`` edge to target the anchored hash, so an agent whose
``constitution_hash`` is current but whose edge still points at an ancient
anchor (the 2026-07-18 incident: a historical pre-atomic reanchor updated
property + blob but never repointed the edge) fails closed at boot. This
module therefore inspects the edge set before declaring "unchanged" and
supports a **same-hash repair**: with ``--force`` + a signed artifact it
atomically upserts the correct edge and removes every stale ``governed_by``
target, without touching the (already-correct) hash, blob, RAG index, or
genesis-audit receipt (#2616).

Pre-flight: caller MUST ensure the agent isn't running. SQLite WAL
locking would otherwise corrupt mid-write.

**Which database.** Governance lives in the database the *runtime* reads, not
in the local ``kestrel_prime.db`` file. On a host configured with
``KESTREL_DB_BACKEND=postgres`` those are two different databases: the anchor
holds the birth record (#2871) while every governance read the agent performs
goes to PostgreSQL. Resolving the write target the same way boot does — from
the process environment — is what makes a reanchor land where the agent will
read it (#2890). The local anchor is deliberately *not* rewritten in that
case: it records what the agent was born under, and #2871 replicates it
additively into the runtime database rather than moving it. The birth *record*
is left alone; the file itself is still opened read/write to read this agent's
DID, which checkpoints a WAL anchor. Nothing in-tree hashes or signs the anchor
file, so that is a storage detail, not a governance one.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kestrel_sovereign.constitution.anchored_bytes import (
    read_anchored_constitution,
)
from kestrel_sovereign.constitution.emancipation import (
    EmancipationConfigError,
    EmancipationContract,
    check_iron_rule,
    contract_from_json,
    contract_to_json,
    parse_emancipation_block,
    unwitnessed_emancipation_downgrade,
)
from kestrel_sovereign.constitution.amendment_artifact import (
    AmendmentArtifactError,
    AmendmentArtifactVerification,
    load_verified_reanchor_artifact,
)
from kestrel_sovereign.constitution.reanchor_receipt import (
    supersede_constitution_reanchor,
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


class ReanchorTargetError(Exception):
    """The database the runtime reads cannot be identified or opened."""


@dataclass(frozen=True)
class ReanchorTarget:
    """The database a reanchor writes, whose agent it writes, and what a file
    backup can protect.

    ``anchor_path`` is always the agent's local ``kestrel_prime.db``. It is
    how a directory is known to be an agent (twelve existence checks across
    seven modules — see #2843), where the birth record lives, and — on every
    backend — where this agent's DID is read from. It is the *write* target
    only when the runtime reads it, i.e. on a SQLite host.

    ``agent_did`` is not decoration. A PostgreSQL host holds every local
    agent in one ``graph_nodes`` table, and an unbound
    ``AsyncGraphStore`` scopes to ``1 = 1``, so "the agent node" is whichever
    row the database returns first. Binding is what makes a reanchor of Emma a
    reanchor of Emma.
    """

    anchor_path: Path
    backend: str
    agent_did: str
    dsn: str | None = None

    @property
    def writes_to_anchor(self) -> bool:
        return self.backend == "sqlite"

    def describe(self) -> str:
        if self.writes_to_anchor:
            return f"sqlite:{self.anchor_path}"
        return f"{self.backend}:{_redacted_dsn(self.dsn)}"

    def open_storage(self, *, read_only: bool = False) -> AsyncStorage:
        """Open the runtime database, bound to this agent.

        ``read_only`` is for the inspection paths. Target resolution and drift
        detection run BEFORE the ``if not force:`` early return, and this
        command is documented as performing no write without ``--force`` — but
        it opened read-write regardless. SQLite serialises writers at the file
        level and the per-connection write-unit lock cannot serialise a SECOND
        connection to the same file, so a drift-only inspection could contend
        with a running agent's database. In #2920 that dropped a live agent
        into Safe Mode; the caller saw nothing, because the cost landed on the
        agent (#2920).

        The backend is passed **explicitly** in both branches. ``AsyncStorage``
        falls back to ``KESTREL_DB_BACKEND`` when it is not, which is how a
        SQLite path argument could silently be redirected to PostgreSQL (and
        vice versa) depending on the operator's shell — the ambiguity #2890 is
        about. Deciding once, here, and reporting the decision is the fix.

        ``agent_id`` is passed in both branches too, exactly as boot does
        (``kestrel_agent.py`` binds ``agent_id=self.did`` on SQLite *and*
        PostgreSQL). An offline tool that reads wider than the runtime can
        answer a question about the wrong tenant.
        """
        if self.writes_to_anchor:
            return AsyncStorage(
                str(self.anchor_path),
                backend="sqlite",
                agent_id=self.agent_did,
                read_only=read_only,
            )
        return AsyncStorage(
            backend=self.backend,
            dsn=self.dsn,
            agent_id=self.agent_did,
            read_only=read_only,
        )


def _redacted_dsn(dsn: str | None) -> str:
    """Render a DSN safe to print: scheme, host, and database only.

    Reanchor output is pasted into tickets and CI logs; a DSN carries a
    password. ``urlsplit`` parses lazily — an invalid port raises from
    ``parts.port``, not from the split — so the whole read is guarded.
    """
    if not dsn:
        return "(from environment)"
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(dsn)
        if not parts.hostname:
            return "(non-URL DSN)"
        host = parts.hostname
        if parts.port:
            host = f"{host}:{parts.port}"
        return f"{parts.scheme}://{host}{parts.path}"
    except ValueError:
        return "(unparseable DSN)"


async def resolve_reanchor_target(
    agent_dir: Path,
    *,
    backend: str | None = None,
    dsn: str | None = None,
) -> ReanchorTarget:
    """Resolve the database the *runtime* would open for this agent, and the
    tenant it would bind.

    The backend rule is copied from the host that actually starts these agents
    — ``agent_manager._initialize_agent``:

    .. code-block:: python

        if db_backend.lower() == "postgres" and database_url:

    PostgreSQL **and** a DSN, or SQLite at the anchor. Being stricter than that
    is not caution: a host with ``KESTREL_DB_BACKEND=postgres`` and no DSN runs
    its agents on the local SQLite file, so refusing it would refuse a reanchor
    that ``main`` performs correctly, with no flag to override.

    The DID comes from the local anchor through the same reader the host
    uses (``identity.local_anchor.read_anchor_agent_did``), which refuses a
    directory holding more than one agent root rather than picking by
    incidental row order.

    Raises:
        ReanchorTargetError: Unsupported backend, or the anchor cannot name
            exactly one agent. The anchor is required on *every* backend: it
            is the only artifact that says which tenant this reanchor is for,
            and boot requires it too.
    """
    # ``.lower()`` and nothing else. ``_initialize_agent`` does not strip, so
    # ``KESTREL_DB_BACKEND="postgres "`` starts the agent on SQLite; stripping
    # here would point the reanchor at PostgreSQL instead — #2890 again with
    # the two databases exchanged. Copying the rule means copying it exactly.
    #
    # ``is not None``, not ``or``: an empty string is an *answer*, not a
    # missing argument. A project ``.env`` that blanks ``KESTREL_DATABASE_URL``
    # puts the spawned agent on SQLite, and a caller relaying that answer must
    # not have it fall through to whatever DSN is still exported in the shell —
    # which would send `--force` into an unrelated PostgreSQL while the tool
    # that prescribed it was reading the local file.
    resolved = (
        backend if backend is not None
        else os.environ.get("KESTREL_DB_BACKEND", "sqlite")
    ).lower()
    if resolved not in ("sqlite", "postgres"):
        # The runtime does not validate this either — ``_initialize_agent``
        # tests `== "postgres"` and falls through to SQLite for anything else,
        # so an agent configured `mysql` really is running on the local file
        # and that is what a reanchor must target. Refusing would be stricter
        # than the thing being repaired. Name it rather than fail on it.
        logger.warning(
            "KESTREL_DB_BACKEND=%r is not a backend this runtime supports; "
            "agents configured this way run on the local SQLite anchor, so "
            "that is what will be reanchored.",
            resolved,
        )
        resolved = "sqlite"

    anchor_path = agent_dir / "kestrel_prime.db"
    # Deferred: agent_manager pulls in the whole agent runtime. Reusing its
    # reader rather than re-implementing one keeps identity resolution to a
    # single authority — duplicating it is how two tools come to disagree
    # about who an agent is.
    from kestrel_sovereign.identity.local_anchor import (
        AgentDIDLookupMode,
        read_anchor_agent_did,
    )

    try:
        # INITIALIZATION, which opens the anchor ``mode=rw`` so SQLite can
        # replay a WAL before reading. That is a write — it checkpoints the
        # file and clears its sidecars — during what the caller thinks is a
        # read, and COLD_READ_ONLY would avoid it. But COLD_READ_ONLY *refuses*
        # an anchor with live WAL state, and leftover sidecars after an unclean
        # stop are ordinary. Refusing there would brick the #2616 edge repair
        # for exactly the agents most likely to need it, to protect a
        # file-mtime property nothing depends on. Checkpointing preserves the
        # record; it only settles how it is stored.
        agent_did = await read_anchor_agent_did(
            str(agent_dir), mode=AgentDIDLookupMode.INITIALIZATION
        )
    except ValueError as exc:
        raise ReanchorTargetError(
            f"Cannot identify the agent in {agent_dir}: {exc}. The local "
            f"{anchor_path.name} names the tenant this reanchor is for, on "
            f"every backend."
        ) from exc

    resolved_dsn = (
        dsn if dsn is not None else os.environ.get("KESTREL_DATABASE_URL")
    )
    if resolved == "postgres" and resolved_dsn:
        postgres = ReanchorTarget(
            anchor_path=anchor_path,
            backend="postgres",
            agent_did=agent_did,
            dsn=resolved_dsn,
        )
        return postgres
    return ReanchorTarget(
        anchor_path=anchor_path, backend="sqlite", agent_did=agent_did
    )


@dataclass(frozen=True)
class ReanchorResult:
    """Outcome of :func:`reanchor_constitution`.

    Exactly one of ``unchanged`` / ``drift_unforced`` / ``reanchored`` /
    ``error`` describes the outcome. ``iron_rule_violation`` is a *label* on
    an error, not a fourth outcome: it marks the subset of refusals that are a
    #1118 transgression rather than the guard being unable to decide.
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
    #: When set, ``error`` is a #1118 Iron Rule refusal — the candidate really
    #: would narrow or revoke an active Emancipation Contract. Deliberately
    #: *not* set for refusals where the guard could not decide (unreadable
    #: anchored bytes, an ambiguous Amendment VIII): those are also errors, but
    #: calling them violations misnames what happened.
    iron_rule_violation: str | None = None
    #: True when pre-write inspection found the ``governed_by`` edge set
    #: inconsistent with the expected anchor — missing, mis-targeted, or
    #: carrying stale extra targets (the 2026-07-18 incident shape, #2616).
    #: Set alongside ``drift_unforced`` / ``reanchored`` so the CLI can
    #: explain an edge-only (same-hash) repair.
    governance_edge_drift: bool = False
    #: ``governed_by`` targets that did not match the expected anchor at
    #: inspection time. Informational — the write path re-reads the edge
    #: set inside its transaction before deleting anything.
    stale_edge_targets: tuple[str, ...] = ()
    #: The database this run read and (on a forced run) wrote — ``sqlite`` or
    #: ``postgres``. ``db_path`` is the local anchor either way, so this is
    #: what says whether the write landed where the runtime reads (#2890).
    target_backend: str = "sqlite"
    #: Human-readable target, DSN credentials redacted. Printed by the CLI so
    #: an operator can see which database a reanchor actually touched.
    target_label: str = ""
    #: Set when no file-level backup was taken because the write target is not
    #: a local file. The inner transaction still makes the write atomic; what
    #: is absent is the *outer* net, and an operator relying on it must know.
    backup_unavailable_reason: str | None = None


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
    runtime_backend: str | None = None,
    runtime_dsn: str | None = None,
) -> ReanchorResult:
    """Reanchor one agent to the current canonical constitution.

    Read order matters: we open the DB twice. First (read-only) to
    discover whether drift exists; if no drift OR no ``--force``, we
    return without ever taking the destructive backup. Only when we're
    actually going to write do we copy the DB aside and reopen the
    storage layer for mutation.

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
    try:
        target = await resolve_reanchor_target(
            agent_dir,
            backend=runtime_backend,
            dsn=runtime_dsn,
        )
    except ReanchorTargetError as exc:
        return ReanchorResult(
            agent_name=agent_name,
            db_path=db_path,
            canonical_path=canonical_path,
            old_hash=None,
            new_hash=None,
            backup_path=None,
            error=str(exc),
        )

    def _result(**kwargs) -> ReanchorResult:
        """Stamp every outcome with the database it describes.

        A reanchor result that does not say which database it read is the
        defect #2890 is about: the same command, the same output, two
        different databases depending on the host's configuration.
        """
        kwargs.setdefault("agent_name", agent_name)
        kwargs.setdefault("db_path", db_path)
        kwargs.setdefault("canonical_path", canonical_path)
        kwargs.setdefault("backup_path", None)
        return ReanchorResult(
            target_backend=target.backend,
            target_label=target.describe(),
            **kwargs,
        )

    # Pre-flight the canonical source so an unreadable path returns a clean
    # ReanchorResult error rather than blowing up inside the resolver below.
    try:
        canonical_path.read_bytes()
    except OSError as exc:
        return _result(
            old_hash=None,
            new_hash=None,
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
        return _result(
            old_hash=None,
            new_hash=None,
            error=(
                f"Refusing to reanchor to non-authoritative constitution source "
                f"{canonical_path}: the periodic integrity audit recomputes from "
                f"the packaged governing source, so an agent anchored elsewhere "
                f"would fail its next audit and Safe-Mode. Reanchor against the "
                f"packaged source (omit --constitution-path) or point "
                f"config.CONSTITUTION_PATH at your authoritative source (#2463)."
            ),
        )

    # Opening the runtime database is the first thing here that can reach the
    # network. A PostgreSQL host that is down, or a DSN with the wrong
    # password, must produce a ReanchorResult error like every other refusal,
    # not a traceback out of the CLI.
    try:
        (
            old_hash,
            agent_did,
            anchored_contract_json,
            governed_by_targets,
            anchored_text,
            anchored_present,
            row_exists,
            visible_edge_targets,
        ) = await _read_agent_anchor(target)
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim to the operator
        logger.exception("Could not read the anchor from %s", target.describe())
        return _result(
            old_hash=None,
            new_hash=None,
            error=(
                f"Could not read this agent's governance from "
                f"{target.describe()}: {exc!r}. Nothing was written."
            ),
        )
    # Three states share ``old_hash=None`` and only one of them retargets.
    # A node that exists and is owned but carries no ``constitution_hash``
    # returns its real DID — retargeting there would write the local file while
    # the runtime node it boots from stayed unanchored. A row that exists
    # *unowned* returns an empty DID like a missing one, but replication cannot
    # repair it either: ``add_node`` will not overwrite a foreign-owned row, so
    # sending the repair to SQLite would leave PostgreSQL unreadable and the
    # agent unbootable. Only a physically absent row is the state first-boot
    # replication fixes, so only that one retargets.
    if not target.writes_to_anchor and await runtime_record_is_pending(target):
        # PostgreSQL has nothing for this agent. Boot does not fail there: it
        # copies the birth record out of the local anchor (#2871) and audits
        # *that*, so the bytes that will govern this agent at its next start
        # are the anchor's — and the anchor is what a pre-boot repair has to
        # change. Reporting "no constitution_hash" here would leave a stale
        # anchor to safe-mode the agent, and `kestrel doctor` prescribes this
        # very command for exactly that drift: a remedy that cannot clear the
        # finding that prescribed it. Doctor reads the anchor in this state for
        # the same reason (#2892); the two have to mean the same bytes.
        #
        # Decided from the read that already happened rather than a probe
        # before it: an extra pre-flight connection paid the whole timeout
        # twice on an unreachable database, and did it before the canonical
        # file had even been read.
        logger.info(
            "%s holds no record for %s; retargeting the local anchor, which is "
            "what first boot will replicate and audit.",
            target.describe(), target.agent_did,
        )
        target = ReanchorTarget(
            anchor_path=target.anchor_path,
            backend="sqlite",
            agent_did=target.agent_did,
        )
        try:
            (
                old_hash,
                agent_did,
                anchored_contract_json,
                governed_by_targets,
                anchored_text,
                anchored_present,
                row_exists,
                visible_edge_targets,
            ) = await _read_agent_anchor(target)
        except Exception as exc:  # noqa: BLE001 — surfaced to the operator
            logger.exception("Could not read the anchor at %s", target.describe())
            return _result(
                old_hash=None,
                new_hash=None,
                error=(
                    f"Could not read this agent's governance from "
                    f"{target.describe()}: {exc!r}. Nothing was written."
                ),
            )

    if old_hash is None:
        return _result(
            old_hash=None,
            new_hash=None,
            error=(
                f"Agent has no constitution_hash property in {target.describe()}. "
                "Re-incept the agent rather than reanchoring."
            ),
        )

    # --- #1118: Iron Rule + active-form re-application -------------------
    try:
        anchored_contract = contract_from_json(anchored_contract_json)
    except EmancipationConfigError as exc:
        return _result(
            old_hash=old_hash, new_hash=None,
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
            return _result(
                old_hash=old_hash, new_hash=None,
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
        return _result(
            old_hash=old_hash, new_hash=None,
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
    # Every other refusal in here returns a ReanchorResult; ``cli.py`` calls
    # this bare inside ``asyncio.run``, so anything that escapes is a traceback
    # at an operator. The resolver is documented to raise so its callers fail
    # closed — including ``AmbiguousAmendmentVIII`` for a governing source with
    # two Amendment VIII headings — and failing closed here means saying so.
    try:
        new_content = resolve_governing_constitution_bytes(
            effective_contract if (
                effective_contract is not None and effective_contract.enabled
            ) else None,
            constitution_path=str(canonical_path),
        )
        new_text = new_content.decode("utf-8")
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return _result(
            old_hash=old_hash, new_hash=None,
            error=(
                f"Could not resolve the governing constitution from "
                f"{canonical_path}: {exc}. Nothing was written."
            ),
        )

    new_hash = hashlib.sha256(new_content).hexdigest()

    # #2465: the Iron Rule for an agent with NO structured receipt. The
    # backfill below only fires when a ``kestrel.toml [emancipation]`` block
    # supplies a candidate; with no block, ``anchored_contract`` is None, the
    # resolver above just rendered the dormant canonical text, and a
    # Sovereign-signed artifact over those exact bytes would authorize erasing
    # the authored terms. The anchored bytes are the contract when nothing
    # else witnesses it, so the only permitted reanchor is one that reproduces
    # their Amendment VIII section. Shared with the live command so the two
    # entry points cannot diverge on this.
    downgrade = unwitnessed_emancipation_downgrade(
        anchored_contract=anchored_contract,
        anchored_text=anchored_text,
        anchored_present=anchored_present,
        old_hash=old_hash,
        new_hash=new_hash,
        new_text=new_text,
    )
    if downgrade is not None:
        return _result(
            old_hash=old_hash, new_hash=new_hash,
            error=downgrade.message,
            # Only stamp it when it IS one. Unreadable bytes and an ambiguous
            # Amendment VIII are the guard unable to decide, not a Sovereign
            # transgression, and reporting them as a violation sends an
            # operator hunting for something that is not there.
            iron_rule_violation=(
                downgrade.message if downgrade.iron_rule_violation else None
            ),
        )

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

    # Edge consistency is part of the drift decision (#2616). Proof 2 of the
    # fail-closed integrity audit requires an ``agent --governed_by--> anchor``
    # edge targeting the anchored hash, so a matching hash alone does NOT mean
    # "nothing to do": an edge left on an ancient anchor by a historical
    # pre-atomic reanchor still safe-modes the agent at boot. Extra stale
    # targets alongside the correct edge don't fail proof 2, but they mean the
    # agent nominally has two governing constitutions — repair those too.
    stale_edge_targets = tuple(t for t in governed_by_targets if t != new_hash)
    # Proof 2 is judged on what the runtime can see, not on what physically
    # exists: an edge at the right hash whose ownership witness is missing is
    # invisible to the bound store the audit reads through, so the agent
    # safe-modes while the row sits there looking correct.
    governance_edge_drift = (
        new_hash not in visible_edge_targets or bool(stale_edge_targets)
    )

    if (
        old_hash == new_hash
        and not needs_sidecar_backfill
        and not governance_edge_drift
    ):
        return _result(
            old_hash=old_hash,
            new_hash=new_hash,
            unchanged=True,
        )

    if not force:
        return _result(
            old_hash=old_hash,
            new_hash=new_hash,
            drift_unforced=True,
            governance_edge_drift=governance_edge_drift,
            stale_edge_targets=stale_edge_targets,
        )

    # Authorization is a pre-write gate. The graph DB is the object being
    # protected, so neither its root properties nor any material derived from
    # them is consulted here (#2499). Complete root resolution, artifact IO,
    # and signature verification before even taking the backup.
    if amendment_artifact_path is None:
        return _result(
            old_hash=old_hash,
            new_hash=new_hash,
            error=(
                "A Sovereign-signed amendment artifact is required for a "
                "forced reanchor. Pass --signed-artifact and configure the "
                "external Sovereign trust root."
            ),
        )

    try:
        trusted_did_document = load_sovereign_trust_root(
            explicit_path=sovereign_trust_root_path,
            agent_dids={agent_did},
        )
        (
            amendment_artifact_bytes,
            amendment_artifact,
            amendment_verification,
        ) = load_verified_reanchor_artifact(
            amendment_artifact_path,
            trusted_did_document=trusted_did_document,
            expected_constitution_sha256=new_hash,
        )
    except (SovereignTrustRootError, AmendmentArtifactError) as exc:
        return _result(
            old_hash=old_hash,
            new_hash=new_hash,
            error=str(exc),
        )

    # The pre-write refusal that used to sit here is gone with #2893. It
    # existed because a fleet-wide artifact is ONE content-addressed node id
    # and ``add_node`` would not admit a foreign-owned one, so the second agent
    # to anchor the same signed file failed mid-write; refusing early at least
    # named the artifact and a workaround. The node now carries only fields
    # derived from the artifact bytes, so two agents anchoring the same file
    # co-own one identical row — the same rule the constitution document has
    # always followed. There is nothing left to refuse.

    # The file-level backup is the OUTER safety net; the write transaction is
    # the inner one. A PostgreSQL runtime has no file to copy, and copying the
    # local anchor there would be worse than taking none — it would name a
    # backup of a database this write does not touch. Say so instead: the
    # write is still atomic, but restoring a *successful but unwanted*
    # reanchor is the operator's pg_dump, not ours.
    backup_path: Path | None = None
    backup_unavailable_reason: str | None = None
    if target.writes_to_anchor:
        backup_path = _backup_db(db_path)
        logger.info("Backed up agent DB to %s before reanchor", backup_path)
    else:
        backup_unavailable_reason = (
            f"no file-level backup: governance for this agent lives in "
            f"{target.describe()}, not in a local file. The write below is a "
            f"single transaction and rolls back on failure; take a database "
            f"snapshot first if you want to be able to undo a successful one."
        )
        logger.warning(
            "Reanchoring %s against %s with no file-level backup.",
            agent_name, target.describe(),
        )

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
        await _write_reanchor(
            target=target,
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
        logger.exception(
            "Reanchor against %s failed; backup: %s",
            target.describe(), backup_path or "(none)",
        )
        recovery = (
            f" DB backup at {backup_path}"
            if backup_path is not None
            else " The write was one transaction and rolled back; no file "
                 "backup was taken because the target is not a local file."
        )
        return _result(
            old_hash=old_hash,
            new_hash=new_hash,
            backup_path=backup_path,
            backup_unavailable_reason=backup_unavailable_reason,
            error=f"Reanchor failed mid-write ({exc!r}).{recovery}",
        )

    return _result(
        old_hash=old_hash,
        new_hash=new_hash,
        backup_path=backup_path,
        backup_unavailable_reason=backup_unavailable_reason,
        reanchored=True,
        governance_edge_drift=governance_edge_drift,
        stale_edge_targets=stale_edge_targets,
    )


async def runtime_record_is_pending(target: ReanchorTarget) -> bool:
    """Whether first boot has still to put this agent into the runtime database.

    One rule, asked the same way by every tool that has to decide *which bytes
    will govern this agent next*: before replication the answer is the local
    anchor, because that is what boot copies and then audits (#2871). Doctor
    reports on the anchor in this state, ``reanchor_constitution`` retargets to
    it, and ``setup.overlay_anchor`` does the same — three tools that must
    agree, so they share this predicate rather than each carrying a copy that
    drifts.

    Pending means **absent or a boot-fabricated placeholder**, and nothing
    else. Those are the two states replication repairs. An *unowned* or
    *foreign-owned* row also reads back empty from a bound store, and looks
    identical from the outside — but ``add_node`` refuses to claim either, so
    boot cannot repair them and redirecting a repair to the anchor would leave
    the runtime broken while reporting success. Those are ledger damage, not
    pending replication, and no tool here can fix them.

    SQLite is never pending: the anchor *is* the runtime database.
    """
    if target.writes_to_anchor:
        return False

    from kestrel_sovereign.identity.birth_record import is_fabricated_placeholder

    async with target.open_storage(read_only=True) as storage:
        # Ownership first, because it can veto both of the states below.
        # ``add_node`` refuses a row owned by anyone other than this agent, and
        # refuses one with several owners, so replication cannot land there —
        # an absent row with a stale foreign witness, or a placeholder a second
        # tenant also claims, is ledger damage wearing the shape of something
        # boot repairs. Calling it pending would send the repair to the anchor
        # and leave PostgreSQL unusable while reporting success.
        owner_rows = await storage.db.fetchall(
            "SELECT agent_id FROM graph_node_owners WHERE node_id = ?",
            (target.agent_did,),
        )
        owners = {row[0] for row in owner_rows}
        if owners - {target.agent_did}:
            return False

        agent = await storage.graph.get_node(target.agent_did)
        if agent is not None:
            return is_fabricated_placeholder(agent, target.agent_did)
        # The bound read found nothing. Only a physically absent row is
        # pending; an existing one this agent cannot see is ledger damage.
        # By ``node_id`` alone, deliberately: a row occupying this DID under
        # some other ``node_type`` still collides on the primary key, and
        # ``add_node`` will refuse it just the same.
        physical = await storage.db.fetchone(
            "SELECT 1 FROM graph_nodes WHERE node_id = ?", (target.agent_did,)
        )
        return physical is None


async def _read_agent_anchor(
    target: ReanchorTarget,
) -> tuple[
    str | None, str, dict | None, tuple[str, ...], str | None, bool, bool,
    tuple[str, ...],
]:
    """Return ``(constitution_hash, agent_did, emancipation_contract_json,
    governed_by_targets, anchored_text, anchored_present, row_exists,
    visible_edge_targets)`` **from the database the runtime reads**.

    Read-only — safe to call before deciding whether to touch the DB.
    Returns ``(None, "", None, (), None, False)`` if the agent node has no
    anchored hash.
    The contract field is ``None`` for dormant agents and for legacy
    agents incepted before #1118 (no JSON receipt was written). The edge
    targets feed the drift decision (#2616): integrity proof 2 requires a
    ``governed_by`` edge at the anchored hash, so the caller must not
    declare "unchanged" on the hash comparison alone.

    ``row_exists`` is the one *unscoped* fact reported here, and it exists to
    keep two different absences apart. The bound read below returns nothing
    both when the row is missing and when it is present without this agent's
    ``graph_node_owners`` witness — and those want opposite handling: a missing
    row is repaired by first-boot replication from the anchor, while an unowned
    row blocks it, because ``add_node`` will not overwrite a foreign-owned row.
    Treating the second as the first sent a forced repair to the local SQLite
    file while PostgreSQL stayed unreadable.

    ``open_storage`` binds every store to ``target.agent_did``, which is what
    stops a PostgreSQL host — one table holding every local agent — from
    answering this question about a neighbour. The lookup is by DID rather
    than ``get_nodes_by_type("agent")[0]`` for the same reason the runtime's
    own ``_get_or_create_agent_node`` is: scoping and naming are different
    guarantees, and only the second one survives an agent that owns more than
    one agent-typed node.
    """
    async with target.open_storage(read_only=True) as storage:
        agent = await storage.graph.get_node(target.agent_did)
        if agent is None or agent.node_type != "agent":
            # Same privileged connection the edge read below uses, and for a
            # related reason: the bound store cannot tell "no row" from "a row
            # this agent does not own".
            # By ``node_id`` alone: a row holding this DID under another
            # ``node_type`` still collides on the primary key, and ``add_node``
            # refuses it just the same, so it is not "absent" in any useful
            # sense.
            physical = await storage.db.fetchone(
                "SELECT 1 FROM graph_nodes WHERE node_id = ?",
                (target.agent_did,),
            )
            return None, "", None, (), None, False, physical is not None, ()
        # Read the governance edges through the privileged maintenance
        # connection, NOT the bound graph store. This repair path exists to
        # heal PRE-LEDGER drift (#2616), and stale edges are unowned by
        # construction — pre-ledger writers left no ``graph_edge_owners`` row —
        # so an ownership-scoped read filters out exactly the edges the repair
        # is looking for. It would report no stale targets, short-circuit to
        # ``unchanged=True``, and leave `doctor` prescribing a `--force`
        # reanchor that answers "nothing to do" while the agent stays
        # safe-moded on integrity proof 2. ``_write_reanchor`` reads the same
        # set the same way and already carries this reasoning.
        #
        # Still tenant-scoped: ``source_id`` is this agent's DID, and no
        # caller-supplied value is interpolated into the SQL.
        edge_rows = await storage.db.fetchall(
            "SELECT target_id FROM graph_edges "
            "WHERE source_id = ? AND label = 'governed_by'",
            (agent.node_id,),
        )
        governed_by_targets = tuple(row[0] for row in edge_rows)

        # And the same edges as the *runtime* sees them. Two reads because
        # there are two questions. "Which stale targets are there to prune"
        # must be unscoped, for the reason above. "Is integrity proof 2
        # satisfied" must be scoped, because that is the read boot performs —
        # and a physical edge at the right hash with no ``graph_edge_owners``
        # witness answers yes to the first and no to the second. Deciding
        # "unchanged" from the unscoped set alone meant a forced reanchor
        # reported nothing to do while the agent kept failing proof 2, so
        # doctor's finding named a repair that could not clear it.
        # ``graph.get_edges(..., direction="out")`` is what
        # ``AsyncStorage.get_edges_from`` calls, and the integrity audit reads
        # through that same bound store.
        visible_edges = await storage.graph.get_edges(
            agent.node_id, direction="out"
        )
        visible_edge_targets = tuple(
            edge.target_id for edge in visible_edges
            if edge.label == "governed_by" and edge.source_id == agent.node_id
        )
        anchored_hash = agent.properties.get("constitution_hash")
        anchored_text: str | None = None
        # ABSENT and UNREADABLE are different answers (#2465), and telling them
        # apart takes the privileged connection for the same reason the edge
        # read above does: ``storage.files`` is bound, so a blob with no
        # ``file_owners`` row reads back as absent — the state of every agent
        # in the cohort this guard protects whose governance edge has drifted.
        # See :mod:`kestrel_sovereign.constitution.anchored_bytes`.
        anchored_present = False
        if anchored_hash:
            anchored_text, anchored_present = await read_anchored_constitution(
                storage.db, anchored_hash
            )
        return (
            anchored_hash,
            agent.node_id,
            agent.properties.get("emancipation_contract"),
            governed_by_targets,
            anchored_text,
            anchored_present,
            True,
            visible_edge_targets,
        )


async def _write_reanchor(
    *,
    target: ReanchorTarget,
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
) -> None:
    """Apply the five governance locations plus authorization atomically.

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
      3. Repair the governed_by edge set: upsert the correct edge first,
         then delete every stale target — so a concurrent reader inside
         the transaction (if any) never sees zero governing
         constitutions. Deleting ALL stale targets (not just
         ``old_hash``) is what heals the 2026-07-18 incident shape,
         where a historical pre-atomic reanchor left the edge on an
         ancient anchor that ``delete_edge(old_hash)`` would never
         touch (#2616). This also makes the writer valid for a
         same-hash (``old_hash == new_hash``) edge repair.
      4. Re-index RAG: chunk the new content, then drop the old chunks
         (same "always have something" reasoning). Skipped entirely in
         a same-hash repair — the chunks for ``new_hash`` ARE the live
         index; re-chunking would duplicate them and "dropping the old"
         would drop the fresh ones.
      5. Update the agent node's properties last — that's the pointer
         everyone reads, so flipping it is the conceptual commit. The
         genesis-audit receipt is superseded only when the hash actually
         moves: receipts are bound to their constitution hash, so a
         same-hash repair leaves the existing receipt valid and must not
         force a needless pending re-audit.

    If any step raises, the context manager rolls back and the DB is
    left in its pre-transaction state. The caller's file-level backup, when
    the target is a local file, remains untouched and available either way.
    """
    async with target.open_storage() as storage:
        storage.graph.bind_agent(agent_did)
        storage.files.bind_agent(agent_did)
        storage.rag.bind_agent(agent_did)
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
            await storage.graph.add_node(
                GraphNode(
                    node_id=artifact_hash,
                    node_type="constitution_amendment_artifact",
                    label="Signed Constitution Reanchor Artifact",
                    # Content-derived fields only. Every one of these is a
                    # *signed* field of the artifact, so two agents anchoring
                    # the same file compute the same node — which is what lets
                    # a shared PostgreSQL hold one row for the fleet (#2893).
                    # ``source_path``, ``anchored_at`` and ``verification`` are
                    # per-agent and live on this agent's own
                    # ``constitution_reanchor`` audit property below.
                    properties={
                        "hash": artifact_hash,
                        "type": "SignedConstitutionAmendment",
                        "artifact_type": amendment_artifact.get("artifact_type"),
                        "constitution_hash": new_hash,
                        "signer": amendment_verification.signer,
                        "created_at": amendment_artifact.get("created_at"),
                    },
                )
            )

            # 3. Repair the governed_by edge set — upsert the correct edge
            # first, then delete every stale target (see docstring).
            # This repair routine explicitly heals pre-ledger governance drift.
            # A bound graph capability intentionally cannot see an unowned
            # legacy edge, so inspect only this exact agent source + label via
            # the privileged maintenance connection.  No target supplied by a
            # caller is interpolated into SQL.
            stale_rows = await storage.db.fetchall(
                "SELECT target_id FROM graph_edges "
                "WHERE source_id = ? AND label = 'governed_by' "
                "AND target_id <> ?",
                (agent_did, new_hash),
            )
            stale_edge_targets = sorted({row[0] for row in stale_rows})

            # A physical edge at the correct hash with no ownership witness is
            # invisible to the bound store — it fails integrity proof 2 — and
            # ``add_edge`` refuses to claim it (``Cannot claim or overwrite an
            # unowned graph edge``), so the whole transaction would roll back
            # and the repair this command was prescribed for could never
            # complete. Drop the witness-less row first and let ``add_edge``
            # lay it down properly, with its ledger entry.
            #
            # Only when it is owned by *nobody*. A row someone else witnesses
            # is not this repair's to delete, and ``add_edge`` refusing there
            # is the correct outcome.
            unwitnessed = await storage.db.fetchall(
                "SELECT 1 FROM graph_edges "
                "WHERE source_id = ? AND target_id = ? AND label = 'governed_by' "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM graph_edge_owners AS owner "
                "  WHERE owner.source_id = graph_edges.source_id "
                "  AND owner.target_id = graph_edges.target_id "
                "  AND owner.label = graph_edges.label)",
                (agent_did, new_hash),
            )
            if unwitnessed:
                logger.info(
                    "Removing an unwitnessed governed_by edge for %s so it can "
                    "be re-created with its ownership row.", agent_did,
                )
                await storage.db.execute(
                    "DELETE FROM graph_edges "
                    "WHERE source_id = ? AND target_id = ? "
                    "AND label = 'governed_by'",
                    (agent_did, new_hash),
                )

            await storage.graph.add_edge(agent_did, new_hash, "governed_by")
            for stale_target in stale_edge_targets:
                await storage.db.execute(
                    "DELETE FROM graph_edge_owners "
                    "WHERE source_id = ? AND target_id = ? "
                    "AND label = 'governed_by'",
                    (agent_did, stale_target),
                )
                await storage.db.execute(
                    "DELETE FROM graph_edges "
                    "WHERE source_id = ? AND target_id = ? "
                    "AND label = 'governed_by'",
                    (agent_did, stale_target),
                )

            # 4. Re-index RAG — only when the governing content actually
            # moved (see docstring for the same-hash hazard).
            if old_hash != new_hash:
                await storage.rag.chunk_document(
                    file_hash=new_hash,
                    content=new_content.decode("utf-8"),
                    chunk_size=500,
                    compute_embeddings=True,
                )
                await storage.rag.delete_chunks_for_file(old_hash)

            # 5. Update the agent's pointer + audit record. By DID, not by
            # "the first agent-typed node": the bind above already scopes this
            # store, but naming the tenant is what makes that a stated
            # invariant rather than a property of the current query.
            agent = await storage.graph.get_node(agent_did)
            if agent is None or agent.node_type != "agent":
                raise RuntimeError("Agent node disappeared mid-reanchor")
            agent.properties["constitution_hash"] = new_hash
            if old_hash != new_hash:
                from kestrel_sovereign.constitution.genesis_audit import (
                    supersede_genesis_audit,
                )

                supersede_genesis_audit(
                    agent.properties,
                    constitution_hash=new_hash,
                    provenance="setup:constitution_reanchor",
                )
            supersede_constitution_reanchor(
                agent.properties,
                receipt={
                    "timestamp": _now_iso(),
                    "old_hash": old_hash,
                    "new_hash": new_hash,
                    "source_path": str(canonical_path),
                    "authorization": authorization,
                    "signed_artifact_hash": artifact_hash,
                    "signed_artifact_path": str(amendment_artifact_path),
                    "signed_artifact_signer": amendment_verification.signer,
                    "signed_artifact_verification": amendment_verification.reason,
                    "stale_edges_removed": stale_edge_targets,
                },
                provenance="setup:constitution_reanchor",
            )
            # Anchor (or refresh) the structured contract receipt.
            # Idempotent for the unchanged-active case; performs the
            # dormant→active activation when reanchor enables Amendment
            # VIII for the first time.
            if emancipation_contract_json is not None:
                agent.properties["emancipation_contract"] = emancipation_contract_json
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
