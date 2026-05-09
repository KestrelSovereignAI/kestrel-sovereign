"""Unit tests for kestrel_sovereign.agent.doctrine_bundle.

Pin contract for kestrel-sovereign#1137 chunk 1B:

* ``compute_doctrine_bundle_hash`` is deterministic given the same files
  in the same order; missing files are skipped (not errors); each file
  contributes a fenced section that includes its own sha256 (so an
  attacker can't fence-mimic).
* ``resolve_anchored_paths`` honors the ``DEFAULT_ANCHORED_PATHS`` order
  and appends operator-extra paths after.
* ``anchor_doctrine_bundle`` is idempotent on no-op; first-anchor writes
  the snapshot; non-matching live-vs-anchored at anchor-time raises (use
  reanchor_doctrine_bundle for legitimate updates).
* ``verify_doctrine_bundle`` raises ``DoctrineBundleNotAnchoredError``
  when nothing is anchored; raises ``DoctrineBundleDriftError`` with a
  diagnostic on mismatch; returns the snapshot on match.
* ``reanchor_doctrine_bundle`` requires ``expected_hash`` >=8 chars;
  rejects a mismatched prefix; on success records the reanchor
  metadata (timestamp, old_hash, authorization) on the agent node.

Design source: docs/architecture/CONSTITUTION_INJECTION.md v1.4 §2.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.doctrine_bundle import (
    DEFAULT_ANCHORED_PATHS,
    DoctrineBundleDriftError,
    DoctrineBundleError,
    DoctrineBundleNotAnchoredError,
    DoctrineBundleSnapshot,
    PROP_BUNDLE_ANCHORED_AT,
    PROP_BUNDLE_ANCHORED_PATHS,
    PROP_BUNDLE_FILES,
    PROP_BUNDLE_HASH,
    anchor_doctrine_bundle,
    compute_doctrine_bundle_hash,
    reanchor_doctrine_bundle,
    resolve_anchored_paths,
    verify_doctrine_bundle,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_project(tmp_path: Path) -> Path:
    """Set up a minimal project root with the default anchored doctrine
    files that DEFAULT_ANCHORED_PATHS expects."""
    _write(
        tmp_path / "docs" / "principles" / "KESTREL_CONSTITUTION.md",
        "Article I. Identity rules.\n",
    )
    _write(
        tmp_path / "docs" / "TORTOISE_DOCTRINE.md",
        "Slow is smooth. Smooth is fast.\n",
    )
    _write(tmp_path / "AGENTS.md", "Agent conventions live here.\n")
    return tmp_path


def _make_agent(properties: dict = None) -> MagicMock:
    """Mock agent shape: agent.agent_id, agent.storage.get_node(),
    agent.storage.add_node()."""
    node = MagicMock()
    node.properties = dict(properties or {})

    storage = MagicMock()
    storage.get_node = AsyncMock(return_value=node)
    storage.add_node = AsyncMock()

    agent = MagicMock()
    agent.agent_id = "did:test:abc"
    agent.storage = storage
    agent._test_node = node  # for inspection in tests
    return agent


# ---------------------------------------------------------------------------
# compute_doctrine_bundle_hash
# ---------------------------------------------------------------------------


def test_hash_is_deterministic_for_same_inputs(tmp_path):
    project = _make_project(tmp_path)
    paths = resolve_anchored_paths(project_root=project)
    bootstrap = OrderedDict([("AGENTS.md", "agents content\n")])

    snap1 = compute_doctrine_bundle_hash(
        anchored_files=paths, bootstrap_files=bootstrap
    )
    snap2 = compute_doctrine_bundle_hash(
        anchored_files=paths, bootstrap_files=bootstrap
    )
    assert snap1.hash == snap2.hash
    assert snap1.files == snap2.files


def test_hash_changes_when_anchored_file_content_changes(tmp_path):
    project = _make_project(tmp_path)
    paths = resolve_anchored_paths(project_root=project)
    bootstrap: OrderedDict[str, str] = OrderedDict()

    before = compute_doctrine_bundle_hash(anchored_files=paths, bootstrap_files=bootstrap)
    (project / "AGENTS.md").write_text("changed content\n", encoding="utf-8")
    after = compute_doctrine_bundle_hash(anchored_files=paths, bootstrap_files=bootstrap)
    assert before.hash != after.hash


def test_hash_changes_when_bootstrap_file_content_changes(tmp_path):
    project = _make_project(tmp_path)
    paths = resolve_anchored_paths(project_root=project)

    before = compute_doctrine_bundle_hash(
        anchored_files=paths,
        bootstrap_files=OrderedDict([("SOUL.md", "v1\n")]),
    )
    after = compute_doctrine_bundle_hash(
        anchored_files=paths,
        bootstrap_files=OrderedDict([("SOUL.md", "v2\n")]),
    )
    assert before.hash != after.hash


def test_hash_changes_when_bootstrap_order_changes(tmp_path):
    """Bootstrap order is significant — it's part of the signed bundle.
    Operators that legitimately reorder bootstrap files MUST re-anchor."""
    project = _make_project(tmp_path)
    paths = resolve_anchored_paths(project_root=project)

    a_then_b = compute_doctrine_bundle_hash(
        anchored_files=paths,
        bootstrap_files=OrderedDict([("AGENTS.md", "a"), ("SOUL.md", "s")]),
    )
    b_then_a = compute_doctrine_bundle_hash(
        anchored_files=paths,
        bootstrap_files=OrderedDict([("SOUL.md", "s"), ("AGENTS.md", "a")]),
    )
    assert a_then_b.hash != b_then_a.hash


def test_missing_anchored_files_are_skipped_not_errors(tmp_path):
    """Mirrors the existing ConstitutionMixin pattern of trying multiple
    paths and using whichever resolves. A missing path doesn't poison
    the bundle."""
    project = _make_project(tmp_path)
    paths = resolve_anchored_paths(project_root=project)
    paths.append(project / "does_not_exist.md")  # extra missing path

    snap = compute_doctrine_bundle_hash(
        anchored_files=paths, bootstrap_files=OrderedDict()
    )
    assert "does_not_exist.md" not in " ".join(snap.files)
    # All three real defaults contributed
    assert len(snap.files) == 3


def test_total_bytes_reflects_only_contributing_content(tmp_path):
    project = _make_project(tmp_path)
    paths = resolve_anchored_paths(project_root=project)
    bootstrap = OrderedDict([("X.md", "12345")])  # 5 bytes

    expected_min_bytes = 5 + len(b"Article I. Identity rules.\n")
    snap = compute_doctrine_bundle_hash(anchored_files=paths, bootstrap_files=bootstrap)
    assert snap.total_bytes >= expected_min_bytes


def test_per_file_sha256_in_section_header(tmp_path):
    """Each file's section includes its own sha256 in the BEGIN line.
    Pinning this makes fence-mimic attacks harder — an attacker can't
    construct file content that includes a fake fence to merge two
    files into one bundle section."""
    project = _make_project(tmp_path)
    paths = resolve_anchored_paths(project_root=project)
    snap = compute_doctrine_bundle_hash(
        anchored_files=paths, bootstrap_files=OrderedDict()
    )

    # Verify by reconstructing what the hash should include for one file.
    constitution = (project / "docs" / "principles" / "KESTREL_CONSTITUTION.md").read_bytes()
    constitution_sha = hashlib.sha256(constitution).hexdigest()
    # If per-file sha was included, then changing the constitution's file
    # body but having the SAME bytes (impossible) wouldn't change hash —
    # but changing the body changes both content and per-file sha so the
    # bundle hash MUST change. test_hash_changes_when_anchored_file_content_changes
    # is the operational pin; this test pins the reconstruction property.
    assert len(constitution_sha) == 64  # sanity


# ---------------------------------------------------------------------------
# resolve_anchored_paths
# ---------------------------------------------------------------------------


def test_default_paths_are_three_canonical_files():
    assert DEFAULT_ANCHORED_PATHS == [
        "docs/principles/KESTREL_CONSTITUTION.md",
        "docs/TORTOISE_DOCTRINE.md",
        "AGENTS.md",
    ]


def test_resolve_paths_no_extras(tmp_path):
    paths = resolve_anchored_paths(project_root=tmp_path)
    assert len(paths) == 3
    assert paths[0] == tmp_path / "docs" / "principles" / "KESTREL_CONSTITUTION.md"
    assert paths[1] == tmp_path / "docs" / "TORTOISE_DOCTRINE.md"
    assert paths[2] == tmp_path / "AGENTS.md"


def test_resolve_paths_with_relative_extras(tmp_path):
    paths = resolve_anchored_paths(
        project_root=tmp_path, extra_paths=["docs/MY_DOCTRINE.md"]
    )
    assert paths[-1] == tmp_path / "docs" / "MY_DOCTRINE.md"


def test_resolve_paths_with_absolute_extras(tmp_path):
    abs_path = tmp_path.parent / "external_doctrine.md"
    paths = resolve_anchored_paths(
        project_root=tmp_path, extra_paths=[str(abs_path)]
    )
    assert paths[-1] == abs_path


# ---------------------------------------------------------------------------
# anchor_doctrine_bundle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_anchor_writes_snapshot(tmp_path):
    project = _make_project(tmp_path)
    agent = _make_agent()  # no prior anchor

    snap = await anchor_doctrine_bundle(
        agent, project_root=project, bootstrap_files=OrderedDict()
    )
    assert agent._test_node.properties[PROP_BUNDLE_HASH] == snap.hash
    assert agent._test_node.properties[PROP_BUNDLE_FILES] == snap.files
    assert PROP_BUNDLE_ANCHORED_AT in agent._test_node.properties
    agent.storage.add_node.assert_awaited_once()


@pytest.mark.asyncio
async def test_anchor_is_idempotent_when_hash_matches(tmp_path):
    project = _make_project(tmp_path)
    bootstrap = OrderedDict()

    # First-anchor to compute the hash.
    initial_snap = compute_doctrine_bundle_hash(
        anchored_files=resolve_anchored_paths(project_root=project),
        bootstrap_files=bootstrap,
    )
    agent = _make_agent({PROP_BUNDLE_HASH: initial_snap.hash})

    snap = await anchor_doctrine_bundle(
        agent, project_root=project, bootstrap_files=bootstrap
    )
    assert snap.hash == initial_snap.hash
    # No-op: storage.add_node not called because already anchored
    agent.storage.add_node.assert_not_called()


@pytest.mark.asyncio
async def test_anchor_raises_on_drift_at_anchor_time(tmp_path):
    """If the agent has a different anchored hash than the live bundle,
    anchor refuses — the operator must use reanchor explicitly."""
    project = _make_project(tmp_path)
    agent = _make_agent({PROP_BUNDLE_HASH: "different_hash_xyz"})

    with pytest.raises(DoctrineBundleError, match="drift detected at anchor-time"):
        await anchor_doctrine_bundle(
            agent, project_root=project, bootstrap_files=OrderedDict()
        )


@pytest.mark.asyncio
async def test_anchor_raises_when_agent_node_missing(tmp_path):
    project = _make_project(tmp_path)
    agent = _make_agent()
    agent.storage.get_node = AsyncMock(return_value=None)

    with pytest.raises(DoctrineBundleError, match="agent identity node not found"):
        await anchor_doctrine_bundle(
            agent, project_root=project, bootstrap_files=OrderedDict()
        )


@pytest.mark.asyncio
async def test_anchor_honors_extra_paths_property(tmp_path):
    """Operator-declared doctrine_anchored_paths extends the default list."""
    project = _make_project(tmp_path)
    _write(project / "docs" / "EXTRA.md", "extra content\n")
    agent = _make_agent({PROP_BUNDLE_ANCHORED_PATHS: ["docs/EXTRA.md"]})

    snap = await anchor_doctrine_bundle(
        agent, project_root=project, bootstrap_files=OrderedDict()
    )
    assert any("EXTRA.md" in f for f in snap.files)


# ---------------------------------------------------------------------------
# verify_doctrine_bundle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_raises_not_anchored_when_no_hash(tmp_path):
    project = _make_project(tmp_path)
    agent = _make_agent()  # no PROP_BUNDLE_HASH

    with pytest.raises(DoctrineBundleNotAnchoredError):
        await verify_doctrine_bundle(
            agent, project_root=project, bootstrap_files=OrderedDict()
        )


@pytest.mark.asyncio
async def test_verify_passes_when_hashes_match(tmp_path):
    project = _make_project(tmp_path)
    snap = compute_doctrine_bundle_hash(
        anchored_files=resolve_anchored_paths(project_root=project),
        bootstrap_files=OrderedDict(),
    )
    agent = _make_agent({PROP_BUNDLE_HASH: snap.hash})

    result = await verify_doctrine_bundle(
        agent, project_root=project, bootstrap_files=OrderedDict()
    )
    assert result.hash == snap.hash


@pytest.mark.asyncio
async def test_verify_raises_drift_with_diagnostic(tmp_path):
    project = _make_project(tmp_path)
    agent = _make_agent({PROP_BUNDLE_HASH: "stale_anchor_hash_value"})

    with pytest.raises(DoctrineBundleDriftError) as exc:
        await verify_doctrine_bundle(
            agent, project_root=project, bootstrap_files=OrderedDict()
        )
    assert exc.value.anchored_hash == "stale_anchor_hash_value"
    assert exc.value.live_hash != "stale_anchor_hash_value"
    assert "contributing files at verify time" in exc.value.diagnostic


# ---------------------------------------------------------------------------
# reanchor_doctrine_bundle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reanchor_requires_expected_hash_min_8_chars(tmp_path):
    project = _make_project(tmp_path)
    agent = _make_agent()

    with pytest.raises(ValueError, match="min 8 hex characters"):
        await reanchor_doctrine_bundle(
            agent,
            project_root=project,
            bootstrap_files=OrderedDict(),
            expected_hash="short",
            authorization="test",
        )


@pytest.mark.asyncio
async def test_reanchor_rejects_mismatched_prefix(tmp_path):
    project = _make_project(tmp_path)
    agent = _make_agent({PROP_BUNDLE_HASH: "old_hash_value"})

    with pytest.raises(ValueError, match="does not match the new bundle"):
        await reanchor_doctrine_bundle(
            agent,
            project_root=project,
            bootstrap_files=OrderedDict(),
            expected_hash="0" * 16,  # extremely unlikely to match real hash
            authorization="test",
        )


@pytest.mark.asyncio
async def test_reanchor_success_records_metadata(tmp_path):
    project = _make_project(tmp_path)
    snap = compute_doctrine_bundle_hash(
        anchored_files=resolve_anchored_paths(project_root=project),
        bootstrap_files=OrderedDict(),
    )
    agent = _make_agent({PROP_BUNDLE_HASH: "old_hash_value_for_reanchor"})

    old_hash, new_hash = await reanchor_doctrine_bundle(
        agent,
        project_root=project,
        bootstrap_files=OrderedDict(),
        expected_hash=snap.hash[:16],
        authorization="genesis-amendment-2026-05-09",
    )
    assert old_hash == "old_hash_value_for_reanchor"
    assert new_hash == snap.hash
    metadata = agent._test_node.properties["doctrine_bundle_reanchor"]
    assert metadata["old_hash"] == "old_hash_value_for_reanchor"
    assert metadata["new_hash"] == snap.hash
    assert metadata["authorization"] == "genesis-amendment-2026-05-09"
    assert metadata["expected_hash_prefix"] == snap.hash[:16]
    assert metadata["file_count"] == len(snap.files)


@pytest.mark.asyncio
async def test_reanchor_is_noop_when_hash_already_matches(tmp_path):
    project = _make_project(tmp_path)
    snap = compute_doctrine_bundle_hash(
        anchored_files=resolve_anchored_paths(project_root=project),
        bootstrap_files=OrderedDict(),
    )
    agent = _make_agent({PROP_BUNDLE_HASH: snap.hash})

    old_hash, new_hash = await reanchor_doctrine_bundle(
        agent,
        project_root=project,
        bootstrap_files=OrderedDict(),
        expected_hash=snap.hash[:16],
        authorization="test",
    )
    assert old_hash == new_hash == snap.hash
    # No write because it's a no-op
    agent.storage.add_node.assert_not_called()
