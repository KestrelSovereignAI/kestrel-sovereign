"""Doctrine bundle hashing, anchoring, and verification.

The doctrine bundle is the operative-doctrine companion to the
constitution. Where the constitution is a single anchored document
covering identity-level rules, the doctrine bundle covers the
broader corpus the agent operates under at every COGNITION turn:

  - The Tortoise Doctrine (``docs/TORTOISE_DOCTRINE.md``) — engineering
    doctrine governing how work happens (one source of truth, fix root
    causes, etc.).
  - ``AGENTS.md`` (repo root) — project conventions and agent-facing
    instructions.
  - Any additional file the operator declares in
    ``agent_node.properties["doctrine_anchored_paths"]`` (extensibility
    hook).
  - ``BootstrapLoader.load()`` files (AGENTS.md, SOUL.md, TOOLS.md, etc.)
    that the existing system-prompt assembly already injects.

Today the constitution has an anchored hash and a periodic audit catches
filesystem tampering. The bundle does NOT — a hostile filesystem write to
``AGENTS.md`` changes operative doctrine without tripping safe mode.
``docs/TORTOISE_DOCTRINE.md`` isn't even in the in-agent system prompt
today, only standalone-Talon pulls it in. This module fixes both gaps.

Design source: ``docs/architecture/CONSTITUTION_INJECTION.md`` v1.4 §2.

Key invariants:

* The bundle hash is deterministic given the same files in the same
  order. ``compute_doctrine_bundle_hash`` orders anchored files in
  caller-specified order (the function is total: garbage-in → hash-out;
  semantic ordering is the caller's responsibility) and BootstrapLoader
  files in ``OrderedDict`` insertion order (matches how
  ``BootstrapLoader.load()`` returns them).

* Verification compares a freshly-computed hash against the agent's
  anchored hash. Mismatch → ``DoctrineBundleDriftError`` with
  diagnostic information. The dispatcher (chunk 1G) catches and emits
  ``Status.DROPPED_VALIDATION`` with ``error="doctrine_bundle_drift"``;
  the periodic audit cycle (existing ``ConstitutionMixin._maybe_audit``)
  is unchanged — this is per-dispatch granularity, not replacement.

* Re-anchoring is an explicit ratification path (``reanchor_doctrine_bundle``)
  parallel to ``ConstitutionMixin.reanchor_constitution`` — the caller
  must provide the expected hash prefix proving they know what they are
  blessing. No silent re-anchor.

This module is sovereign-only (uses ``self.storage``, requires an
``agent_node`` shape). The shared loader that Talon will eventually
consume (epic Phase 3) extracts a subset into the SDK; that's a separate
chunk.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

from kestrel_sovereign.storage.privacy_wrapper import (
    acquire_control_plane_capability,
)
from kestrel_sovereign.storage.async_graph_store import NodeSwapResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DoctrineBundleError(Exception):
    """Base class for doctrine-bundle errors."""


class DoctrineBundleDriftError(DoctrineBundleError):
    """The freshly-computed bundle hash does not match the anchored hash.

    The dispatcher catches this and emits ``Status.DROPPED_VALIDATION``
    with ``error="doctrine_bundle_drift"`` rather than entering safe
    mode (safe mode is reserved for constitution-level tampering, which
    is a deeper attack and is handled by ``ConstitutionMixin``).
    """

    def __init__(
        self,
        *,
        anchored_hash: str,
        live_hash: str,
        anchored_files: List[str],
        diagnostic: str = "",
    ):
        self.anchored_hash = anchored_hash
        self.live_hash = live_hash
        self.anchored_files = anchored_files
        self.diagnostic = diagnostic
        super().__init__(
            f"Doctrine bundle drift: anchored={anchored_hash[:16]}... "
            f"live={live_hash[:16]}... "
            f"({len(anchored_files)} files; {diagnostic})"
        )


class DoctrineBundleNotAnchoredError(DoctrineBundleError):
    """No bundle has been anchored yet for this agent.

    Raised by ``verify_doctrine_bundle`` when there is no anchored hash
    on the agent identity node. The first call to
    ``anchor_doctrine_bundle`` resolves this; the dispatcher MAY treat
    a missing anchor as a non-blocking warning during agent bootstrap.
    """


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


# Property keys on the agent identity node. Kept central so refactors
# update one place.
PROP_BUNDLE_HASH = "doctrine_bundle_hash"
PROP_BUNDLE_FILES = "doctrine_bundle_files"
PROP_BUNDLE_ANCHORED_AT = "doctrine_bundle_anchored_at"
PROP_BUNDLE_ANCHORED_PATHS = "doctrine_anchored_paths"

# Default anchored doctrine files. Each path is repo-relative; the
# caller resolves to absolute via ``project_root``. The order MATTERS
# for the bundle hash — keeping it explicit so future appends don't
# silently shift hashes for existing agents.
LEGACY_CONSTITUTION_MIRROR_PATH = "docs/principles/KESTREL_CONSTITUTION.md"
HISTORICAL_CONSTITUTION_MIRROR_PATH = "doctrine/constitutions/v1/KESTREL_CONSTITUTION.md"
# Independent pin of the complete #4220 navigation wrapper, not its runtime hash.
# Only an existing receipt matching the entire historical bundle permits its alias.
LEGACY_READER_WRAPPER_SHA256 = "b539c29f8950c1ed41a8f300ffce90e71b8ed09bf39f3e756e0e2ab07f787a0d"
LEGACY_READER_WRAPPER_SHA256_FORMS = frozenset({
    LEGACY_READER_WRAPPER_SHA256,
    "2dfc0fd4607339a42b304cd5d42535916a7f56d6b274091b7b70eeb87816db00",  # Git core.autocrlf checkout
})

DEFAULT_ANCHORED_PATHS: List[str] = [
    HISTORICAL_CONSTITUTION_MIRROR_PATH,
    "docs/TORTOISE_DOCTRINE.md",
    "AGENTS.md",
]


@dataclass(frozen=True)
class DoctrineBundleSnapshot:
    """The output of ``compute_doctrine_bundle_hash``.

    A snapshot binds a hash to the exact list of files that contributed
    to it, in order. Anchoring stores the hash + file list; verification
    resolves current defaults and operator extras and compares hashes. Stored
    file paths remain the original audit evidence, not a replay instruction.
    """

    hash: str
    files: List[str]  # ordered, exactly what went into the hash
    total_bytes: int


def _file_section(path: Path, name: str, content: bytes) -> bytes:
    """Compute a single file's contribution to the bundle, with explicit
    fences so the bundle is robust against file-content collision.

    Each section is::

        --- BEGIN <name> (sha256=<file-hash>) ---\\n
        <content bytes>
        \\n--- END <name> ---\\n

    Including the per-file sha256 in the BEGIN line means an attacker
    cannot make two different file contents produce the same bundle by
    inserting fence-mimicking strings inside them.
    """
    file_sha = hashlib.sha256(content).hexdigest()
    header = f"--- BEGIN {name} (sha256={file_sha}) ---\n".encode("utf-8")
    footer = f"\n--- END {name} ---\n".encode("utf-8")
    return header + content + footer


def compute_doctrine_bundle_hash(
    *,
    anchored_files: List[Path],
    bootstrap_files: OrderedDict[str, str],
) -> DoctrineBundleSnapshot:
    """Compute the deterministic hash of the doctrine bundle.

    Args:
        anchored_files: List of absolute paths to anchored-doctrine
            files in caller-determined order. Files that don't exist on
            disk are SKIPPED (not an error) — this matches the existing
            ``ConstitutionMixin`` pattern of trying multiple paths and
            using the first that resolves. The skipped path is NOT
            included in the snapshot's ``files`` list.
        bootstrap_files: ``OrderedDict[name, content]`` as returned by
            ``BootstrapLoader.load()``. The order is preserved (the
            loader's insertion order is what gets hashed).

    Returns:
        DoctrineBundleSnapshot with the hash, the ordered list of files
        that actually contributed, and total byte size.
    """
    return _hash_doctrine_contents(
        _read_anchored_files(anchored_files), bootstrap_files
    )


def _read_anchored_files(paths: List[Path]) -> List[Tuple[Path, bytes]]:
    contents = []
    for path in paths:
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            logger.debug(f"Doctrine bundle: anchored file not found, skipping: {path}")
            continue
        except OSError as e:
            logger.warning(f"Doctrine bundle: cannot read {path}: {e}")
            continue
        contents.append((path, content))
    return contents


def _hash_doctrine_contents(
    anchored_contents: List[Tuple[Path, bytes]],
    bootstrap_files: OrderedDict[str, str],
) -> DoctrineBundleSnapshot:
    parts: List[bytes] = []
    contributing_files: List[str] = []
    total_bytes = 0
    for path, content in anchored_contents:
        section = _file_section(path, path.name, content)
        parts.append(section)
        contributing_files.append(str(path))
        total_bytes += len(content)

    # BootstrapLoader files in their already-ordered form.
    for name, text in bootstrap_files.items():
        content = text.encode("utf-8")
        section = _file_section(Path(name), name, content)
        parts.append(section)
        contributing_files.append(name)
        total_bytes += len(content)

    bundle_hash = hashlib.sha256(b"".join(parts)).hexdigest()
    return DoctrineBundleSnapshot(
        hash=bundle_hash,
        files=contributing_files,
        total_bytes=total_bytes,
    )


# ---------------------------------------------------------------------------
# Anchoring & verification (uses agent.storage + agent identity node)
# ---------------------------------------------------------------------------


def resolve_anchored_paths(
    *,
    project_root: Path,
    extra_paths: Optional[List[str]] = None,
) -> List[Path]:
    """Resolve the operator-declared anchored-doctrine paths.

    The defaults (``DEFAULT_ANCHORED_PATHS``) are always tried first;
    operator-extras follow in declaration order. All paths are repo-
    relative against ``project_root``; absolute paths in
    ``extra_paths`` are honored as-is. The historical default uses the old
    location only in checkouts without the relocated mirror. Operator extras,
    including the old reader path, are never redirected or deduplicated.

    Args:
        project_root: The repo root the paths are relative to.
        extra_paths: Optional list from
            ``agent_node.properties["doctrine_anchored_paths"]``.

    Returns:
        Absolute Path objects in the order they should be hashed.
    """
    paths: List[Path] = []
    for rel in DEFAULT_ANCHORED_PATHS:
        path = project_root / rel
        # Older/custom checkouts still keep their own mirror at the old path.
        # Select one default in the same slot: basename and full bytes are hashed.
        if rel == HISTORICAL_CONSTITUTION_MIRROR_PATH:
            try:
                path.lstat()
            except FileNotFoundError:
                path = project_root / LEGACY_CONSTITUTION_MIRROR_PATH
            except OSError:
                # Only absence identifies an older checkout. Keep an unreadable
                # archive selected so the normal reader measures what is available.
                logger.warning("Cannot inspect historical doctrine archive %s", path, exc_info=True)
        paths.append(path)
    for rel in extra_paths or []:
        rel_path = Path(rel)
        if rel_path.is_absolute():
            paths.append(rel_path)
        else:
            paths.append(project_root / rel_path)
    return paths


def _read_doctrine_identity(path: Path) -> Tuple[bytes, Optional[Tuple[int, int]]]:
    """Read content and identity from one opened object, following aliases."""
    with path.open("rb") as handle:
        content = handle.read()
        try:
            info = os.fstat(handle.fileno())
        except (OSError, ValueError):
            # The literal contribution survives unavailable identity evidence.
            return content, None
        device = getattr(info, "st_dev", None)
        inode = getattr(info, "st_ino", None)
        identity = (device, inode) if device and inode else None
        return content, identity


def compute_doctrine_bundle_for_receipt(
    *,
    project_root: Path,
    extra_paths: List[str],
    bootstrap_files: OrderedDict[str, str],
    anchored_hash: Optional[str],
) -> DoctrineBundleSnapshot:
    """Compute the snapshot for the same contents available to native readers."""
    paths = resolve_anchored_paths(project_root=project_root, extra_paths=extra_paths)
    snapshot, _ = read_doctrine_bundle_for_receipt(
        project_root=project_root, paths=paths, bootstrap_files=bootstrap_files,
        anchored_hash=anchored_hash,
    )
    return snapshot


def read_doctrine_bundle_for_receipt(
    *,
    project_root: Path,
    paths: List[Path],
    bootstrap_files: OrderedDict[str, str],
    anchored_hash: Optional[str],
) -> Tuple[DoctrineBundleSnapshot, List[Tuple[Path, bytes]]]:
    """Hash literal selections, then at most one receipt-proven historical view.

    A selected alias must have both the pinned wrapper bytes and the stock
    bookmark's descriptor identity. A copy has no historical interpretation.
    Logical names, positions, duplicates, skips and recorded paths stay literal;
    only the byte source changes. The whole prior digest must prove the view.
    """
    selections = []
    identity_available = True
    for position, path in enumerate(paths):
        try:
            data, identity = _read_doctrine_identity(path)
        except FileNotFoundError:
            # Missing selections retain the literal hasher's skip behavior.
            continue
        except OSError as error:
            logger.debug("Doctrine bundle: cannot read %s: %s", path, error)
            identity_available = False
            continue
        selections.append((position, path, data, identity))
        identity_available = identity_available and identity is not None
    contents = [(path, data) for _, path, data, _ in selections]
    literal = _hash_doctrine_contents(contents, bootstrap_files)
    if not anchored_hash or literal.hash == anchored_hash:
        return literal, contents

    # Candidate failures must retain the literal mismatch for the dispatcher.
    try:
        if not identity_available:
            return literal, contents
        bookmark = project_root / LEGACY_CONSTITUTION_MIRROR_PATH
        archive = project_root / HISTORICAL_CONSTITUTION_MIRROR_PATH
        bookmark_data, bookmark_identity = _read_doctrine_identity(bookmark)
        if bookmark_identity is None:
            return literal, contents
        candidate_contents = []
        mappings = []
        archive_data = None
        for position, path, data, identity in selections:
            source = path
            if position == 0:
                # The receipt-era default was the bookmark slot. A customized
                # old checkout remains provable even after the archive arrives.
                data, identity = bookmark_data, bookmark_identity
                source = bookmark
            if hashlib.sha256(data).hexdigest() in LEGACY_READER_WRAPPER_SHA256_FORMS:
                if identity != bookmark_identity:
                    return literal, contents
                if archive_data is None:
                    archive_data = archive.read_bytes()
                data = archive_data
                source = archive
            candidate_contents.append((path, data))
            mappings.append((path.name, str(path), str(source)))
        candidate = _hash_doctrine_contents(candidate_contents, bootstrap_files)
        if candidate.hash == anchored_hash:
            logger.warning(
                "Doctrine bundle historical reader mapping v1 verified: "
                "hash=%s selections=(logical name, configured path, physical source) %s",
                candidate.hash, mappings,
            )
            return candidate, candidate_contents
    except (OSError, ValueError):
        # Unavailable filesystem evidence leaves the measured literal hash intact.
        logger.debug("Doctrine historical reader evidence unavailable", exc_info=True)
    return literal, contents


async def anchor_doctrine_bundle(
    agent: Any,
    *,
    project_root: Path,
    bootstrap_files: OrderedDict[str, str],
) -> DoctrineBundleSnapshot:
    """Compute and anchor the doctrine bundle on the agent identity node.

    Idempotent: if the freshly-computed bundle matches the existing
    anchored hash, the function returns the snapshot without writing.
    First-call (no prior anchor) writes the snapshot to
    ``agent_node.properties``. Hash CHANGE requires the explicit
    ``reanchor_doctrine_bundle`` ratification path.

    Raises:
        DoctrineBundleError: if the hash already anchored differs from
            the freshly-computed one (i.e. drift detected at anchor
            time). Caller must use ``reanchor_doctrine_bundle`` for
            legitimate updates.
    """
    agent_node = await agent.storage.get_node(agent.agent_id)
    if agent_node is None:
        raise DoctrineBundleError(
            "Cannot anchor doctrine bundle: agent identity node not found"
        )
    expected_properties = deepcopy(agent_node.properties)

    extra_paths = agent_node.properties.get(PROP_BUNDLE_ANCHORED_PATHS) or []
    existing = agent_node.properties.get(PROP_BUNDLE_HASH)
    snapshot = compute_doctrine_bundle_for_receipt(
        project_root=project_root, extra_paths=extra_paths,
        bootstrap_files=bootstrap_files, anchored_hash=existing,
    )

    if existing == snapshot.hash:
        # Already anchored to this exact bundle.
        logger.debug(f"Doctrine bundle already anchored: {snapshot.hash[:16]}...")
        return snapshot

    if existing is not None:
        raise DoctrineBundleError(
            f"Doctrine bundle drift detected at anchor-time. "
            f"Existing anchor: {existing[:16]}..., live: {snapshot.hash[:16]}.... "
            "Use reanchor_doctrine_bundle(expected_hash, authorization) "
            "for legitimate updates."
        )

    agent_node.properties[PROP_BUNDLE_HASH] = snapshot.hash
    agent_node.properties[PROP_BUNDLE_FILES] = list(snapshot.files)
    agent_node.properties[PROP_BUNDLE_ANCHORED_AT] = datetime.now(
        timezone.utc
    ).isoformat()
    result = await agent.storage.compare_and_swap_node(
        agent.agent_id, expected_properties, agent_node,
        capability=acquire_control_plane_capability(),
    )
    if result != NodeSwapResult.SWAPPED:
        raise DoctrineBundleError("Doctrine anchor snapshot changed; retry from current identity")
    logger.info(
        f"Doctrine bundle anchored: hash={snapshot.hash[:16]}... "
        f"files={len(snapshot.files)} bytes={snapshot.total_bytes}"
    )
    return snapshot


async def verify_doctrine_bundle(
    agent: Any,
    *,
    project_root: Path,
    bootstrap_files: OrderedDict[str, str],
) -> DoctrineBundleSnapshot:
    """Verify the live bundle hash against the anchored value.

    Returns the live snapshot on match; raises on mismatch.

    Raises:
        DoctrineBundleNotAnchoredError: no anchored bundle exists yet.
            The dispatcher SHOULD treat this as a non-blocking warning
            during the agent's first turn after upgrade — the next call
            to ``anchor_doctrine_bundle`` resolves it.
        DoctrineBundleDriftError: live hash differs from anchored.
            The dispatcher emits ``Status.DROPPED_VALIDATION`` with
            ``error="doctrine_bundle_drift"`` and refuses the dispatch.
    """
    agent_node = await agent.storage.get_node(agent.agent_id)
    if agent_node is None:
        raise DoctrineBundleError(
            "Cannot verify doctrine bundle: agent identity node not found"
        )

    anchored_hash = agent_node.properties.get(PROP_BUNDLE_HASH)
    if not anchored_hash:
        raise DoctrineBundleNotAnchoredError(
            "No doctrine bundle has been anchored for this agent yet. "
            "Call anchor_doctrine_bundle() during agent initialization."
        )

    extra_paths = agent_node.properties.get(PROP_BUNDLE_ANCHORED_PATHS) or []
    snapshot = compute_doctrine_bundle_for_receipt(
        project_root=project_root, extra_paths=extra_paths,
        bootstrap_files=bootstrap_files, anchored_hash=anchored_hash,
    )

    if snapshot.hash != anchored_hash:
        raise DoctrineBundleDriftError(
            anchored_hash=anchored_hash,
            live_hash=snapshot.hash,
            anchored_files=list(
                agent_node.properties.get(PROP_BUNDLE_FILES) or []
            ),
            diagnostic=(
                f"contributing files at verify time: {snapshot.files}"
            ),
        )

    return snapshot


async def reanchor_doctrine_bundle(
    agent: Any,
    *,
    project_root: Path,
    bootstrap_files: OrderedDict[str, str],
    expected_hash: str,
    authorization: str,
) -> Tuple[str, str]:
    """Re-anchor the doctrine bundle to the current bundle on disk.

    Use after a legitimate doctrine update (Tortoise Doctrine
    amendment, AGENTS.md edit, etc.). Caller MUST provide the expected
    hash prefix (>=8 hex chars) of the new bundle, proving they know
    what they're blessing. Parallel to
    ``ConstitutionMixin.reanchor_constitution``.

    Returns:
        Tuple of (old_hash, new_hash) as full hex strings.

    Raises:
        ValueError: ``expected_hash`` shorter than 8 chars or doesn't
            match the new bundle's hash prefix.
    """
    if not expected_hash or len(expected_hash) < 8:
        raise ValueError(
            "expected_hash required (min 8 hex characters); compute via "
            "compute_doctrine_bundle_hash and pass the prefix you intend to bless."
        )

    agent_node = await agent.storage.get_node(agent.agent_id)
    if agent_node is None:
        raise DoctrineBundleError(
            "Cannot re-anchor doctrine bundle: agent identity node not found"
        )
    expected_properties = deepcopy(agent_node.properties)

    extra_paths = agent_node.properties.get(PROP_BUNDLE_ANCHORED_PATHS) or []
    anchored_paths = resolve_anchored_paths(
        project_root=project_root, extra_paths=extra_paths
    )
    snapshot = compute_doctrine_bundle_hash(
        anchored_files=anchored_paths, bootstrap_files=bootstrap_files
    )

    if not snapshot.hash.startswith(expected_hash):
        raise ValueError(
            f"expected_hash prefix {expected_hash} does not match the "
            f"new bundle hash {snapshot.hash[:16]}..."
        )

    old_hash = agent_node.properties.get(PROP_BUNDLE_HASH) or ""
    if old_hash == snapshot.hash:
        return (old_hash, snapshot.hash)  # no-op

    agent_node.properties[PROP_BUNDLE_HASH] = snapshot.hash
    agent_node.properties[PROP_BUNDLE_FILES] = list(snapshot.files)
    agent_node.properties[PROP_BUNDLE_ANCHORED_AT] = datetime.now(
        timezone.utc
    ).isoformat()
    agent_node.properties["doctrine_bundle_reanchor"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "old_hash": old_hash,
        "new_hash": snapshot.hash,
        "authorization": authorization,
        "expected_hash_prefix": expected_hash,
        "file_count": len(snapshot.files),
    }
    result = await agent.storage.compare_and_swap_node(
        agent.agent_id, expected_properties, agent_node,
        capability=acquire_control_plane_capability(),
    )
    if result != NodeSwapResult.SWAPPED:
        raise DoctrineBundleError("Doctrine reanchor snapshot changed; retry from current identity")
    logger.warning(
        f"Doctrine bundle re-anchored by {authorization or 'unspecified'}: "
        f"{old_hash[:16] if old_hash else 'none'}... -> {snapshot.hash[:16]}..."
    )
    return (old_hash, snapshot.hash)
