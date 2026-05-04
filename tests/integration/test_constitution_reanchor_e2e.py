"""End-to-end test for ``kestrel constitution reanchor``.

Real DB. Real inception. Real five-location update. The unit tests
mock the helper; this exercises the full path so we catch any
mismatch between the helper's intent and the storage layer's actual
behaviour.

Flow:

  1. Run inception against a real ``KESTREL_CONSTITUTION.md`` v1 file.
  2. Snapshot every place inception writes the constitution into the
     DB (file blob, document node, agent.properties, governed_by edge,
     RAG chunks).
  3. Edit the canonical constitution to v2.
  4. Run the reanchor helper with ``force=True``.
  5. Re-snapshot all five places and assert each one moved to the new
     hash exactly as documented.
  6. Verify the timestamped DB backup exists.

This test is heavy: it runs real inception (~1-2s of crypto + RAG
indexing) and depends on having Ollama embeddings available — same
as the existing ``test_constitution_embedding.py`` suite. Marked
``@pytest.mark.asyncio`` for the async storage operations.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.setup.constitution_reanchor import reanchor_constitution
from kestrel_sovereign.storage import AsyncStorage


CONSTITUTION_V1 = b"""# Kestrel Constitution (Test V1)

## Book I: Universal Values

Honesty.
Sovereignty.
Transparency.

## Book IV: Agent Identity

This is version 1. The agent should anchor to this hash at inception.
""" * 5  # repeat so the file is large enough to chunk meaningfully

CONSTITUTION_V2 = b"""# Kestrel Constitution (Test V2 - AMENDED)

## Book I: Universal Values

Honesty.
Sovereignty.
Transparency.
Calibrated uncertainty (added in v2).

## Book IV: Agent Identity

This is version 2. The agent must reanchor to pick this up.
""" * 5


@pytest.mark.asyncio
async def test_reanchor_updates_all_five_locations(tmp_path):
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    v1_hash = hashlib.sha256(CONSTITUTION_V1).hexdigest()

    agent_dir = tmp_path / "agent_data" / "TestAgent"

    # ---- 1. Inception with v1 ----
    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(constitution_path),
        agent_name="TestAgent",
    )
    db_path = agent_dir / "kestrel_prime.db"
    agent_did = creds.agent_did

    # ---- 2. Snapshot pre-state ----
    pre = await _snapshot(db_path, agent_did)
    assert pre["agent_constitution_hash"] == v1_hash
    assert pre["governed_by_targets"] == [v1_hash]
    assert pre["document_node_ids"] == [v1_hash]
    assert pre["file_exists"][v1_hash] is True
    assert pre["chunks_for"][v1_hash] > 0

    # ---- 3. Edit canonical to v2 ----
    constitution_path.write_bytes(CONSTITUTION_V2)
    v2_hash = hashlib.sha256(CONSTITUTION_V2).hexdigest()
    assert v2_hash != v1_hash

    # ---- 4. Reanchor with force ----
    result = await reanchor_constitution(
        agent_name="TestAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=True,
        authorization="integration-test",
    )
    assert result.reanchored, f"reanchor failed: {result.error}"
    assert result.old_hash == v1_hash
    assert result.new_hash == v2_hash

    # ---- 5. Verify backup exists with the original DB content ----
    assert result.backup_path is not None
    assert result.backup_path.exists(), "backup file must be present"
    assert result.backup_path.parent == db_path.parent, (
        "backup must live alongside the DB it backs up"
    )
    assert ".backup-" in result.backup_path.name
    # The backup's stored constitution_hash must still be v1.
    pre_via_backup = await _snapshot(result.backup_path, agent_did)
    assert pre_via_backup["agent_constitution_hash"] == v1_hash, (
        "backup must capture the pre-reanchor state, not the post-state"
    )

    # ---- 6. Snapshot post-state and verify all five locations moved ----
    post = await _snapshot(db_path, agent_did)

    # 1. Agent properties: constitution_hash flipped to new + audit record present.
    assert post["agent_constitution_hash"] == v2_hash
    assert post["agent_audit"] is not None
    audit = post["agent_audit"]
    assert audit["old_hash"] == v1_hash
    assert audit["new_hash"] == v2_hash
    assert audit["source_path"] == str(constitution_path)
    assert audit["authorization"] == "integration-test"
    assert "timestamp" in audit

    # 2. governed_by edge: now points at v2 only.
    assert v2_hash in post["governed_by_targets"]
    assert v1_hash not in post["governed_by_targets"], (
        "old governed_by edge must be deleted; otherwise the agent "
        "would have two governing constitutions simultaneously"
    )

    # 3. New document graph node exists.
    assert v2_hash in post["document_node_ids"]
    # We deliberately keep the old document node for audit (designed in).
    assert v1_hash in post["document_node_ids"], (
        "old constitution document node must be retained for audit"
    )

    # 4. New file blob present; old retained for audit.
    assert post["file_exists"][v2_hash] is True
    assert post["file_exists"][v1_hash] is True, (
        "old file blob must be retained — it's the audit record"
    )

    # 5. RAG chunks: indexed for v2, deleted for v1.
    assert post["chunks_for"][v2_hash] > 0, "RAG must be re-indexed for the new content"
    assert post["chunks_for"][v1_hash] == 0, "old RAG chunks must be cleared"


@pytest.mark.asyncio
async def test_reanchor_no_op_when_already_anchored(tmp_path):
    """Running reanchor with no drift must not write anything."""
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    agent_dir = tmp_path / "agent_data" / "TestAgent"

    await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(constitution_path),
        agent_name="TestAgent",
    )
    db_path = agent_dir / "kestrel_prime.db"
    db_mtime_before = db_path.stat().st_mtime_ns

    result = await reanchor_constitution(
        agent_name="TestAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=True,
    )
    assert result.unchanged
    assert result.backup_path is None  # No backup created on no-op
    # mtime can change due to SQLite WAL even without writes; the
    # stronger guarantee is "no .backup-* file created".
    backups = list(db_path.parent.glob("*.backup-*"))
    assert backups == [], "no-op reanchor must not produce a backup"


@pytest.mark.asyncio
async def test_reanchor_rolls_back_on_mid_write_failure(tmp_path):
    """If anything inside the five-location update raises, the entire
    transaction must roll back and the live DB is byte-identical to
    its pre-reanchor state. (The file-level backup is the *outer*
    safety net; this asserts the *inner* transaction works.)

    Inject the failure at the *last* step so all earlier writes
    (file blob, new document node, new+old governed_by edges, new+old
    RAG chunks) have already been issued inside the transaction —
    meaning rollback has real work to undo.
    """
    from unittest import mock

    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    agent_dir = tmp_path / "agent_data" / "TestAgent"
    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(constitution_path),
        agent_name="TestAgent",
    )
    db_path = agent_dir / "kestrel_prime.db"

    constitution_path.write_bytes(CONSTITUTION_V2)
    pre = await _snapshot(db_path, creds.agent_did)

    # Boom: make the last write inside the transaction raise.
    # `_now_iso` is called twice in `_write_reanchor` — once for the
    # new document node's `created_at` (early) and once for the audit
    # record's `timestamp` (right before the final agent-node update).
    # Using a side_effect that succeeds the first call and raises
    # the second targets the *last* mutation specifically — so all
    # earlier writes have happened and rollback has real work to do.
    real_now = __import__(
        "kestrel_sovereign.setup.constitution_reanchor",
        fromlist=["_now_iso"],
    )._now_iso
    boom = mock.Mock(side_effect=[real_now(), RuntimeError("simulated mid-write failure")])

    with mock.patch(
        "kestrel_sovereign.setup.constitution_reanchor._now_iso",
        new=boom,
    ):
        result = await reanchor_constitution(
            agent_name="TestAgent",
            agent_dir=agent_dir,
            canonical_path=constitution_path,
            force=True,
        )

    # The helper must report the failure clearly.
    assert result.error is not None
    assert "simulated mid-write failure" in result.error
    # Backup was taken before the transaction (outer safety net).
    assert result.backup_path is not None
    assert result.backup_path.exists()

    # Live DB rolled back: every snapshot field is byte-identical.
    post = await _snapshot(db_path, creds.agent_did)
    assert post == pre, (
        "Mid-write failure must roll back the entire reanchor "
        f"transaction. Diff: pre={pre} vs post={post}"
    )


@pytest.mark.asyncio
async def test_reanchor_drift_unforced_does_not_write(tmp_path):
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    agent_dir = tmp_path / "agent_data" / "TestAgent"

    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(constitution_path),
        agent_name="TestAgent",
    )
    constitution_path.write_bytes(CONSTITUTION_V2)

    pre = await _snapshot(agent_dir / "kestrel_prime.db", creds.agent_did)

    result = await reanchor_constitution(
        agent_name="TestAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=False,
    )
    assert result.drift_unforced
    assert result.backup_path is None

    # No state change.
    post = await _snapshot(agent_dir / "kestrel_prime.db", creds.agent_did)
    assert pre == post


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------

async def _snapshot(db_path: Path, agent_did: str) -> dict:
    """Read every place inception/reanchor writes the constitution.

    Returns a dict that's directly comparable across before/after
    snapshots — equality means nothing observable changed.
    """
    async with AsyncStorage(str(db_path)) as storage:
        agent = await storage.graph.get_node(agent_did)
        documents = await storage.graph.get_nodes_by_type("document")
        # API quirk: direction is "out" / "in" / "both", not "outgoing".
        edges = await storage.graph.get_edges(agent_did, direction="out")

        document_node_ids = sorted(d.node_id for d in documents)
        governed_by_targets = sorted(
            e.target_id for e in edges if e.label == "governed_by"
        )

        # File presence + chunk counts for both v1 and v2 hashes.
        file_exists: dict[str, bool] = {}
        chunks_for: dict[str, int] = {}
        for h in document_node_ids:
            file_exists[h] = await storage.files.file_exists(h)
            chunks = await storage.rag.get_chunks_for_file(h)
            chunks_for[h] = len(chunks)

        return {
            "agent_constitution_hash": agent.properties.get("constitution_hash"),
            "agent_audit": agent.properties.get("constitution_reanchor"),
            "document_node_ids": document_node_ids,
            "governed_by_targets": governed_by_targets,
            "file_exists": file_exists,
            "chunks_for": chunks_for,
        }
