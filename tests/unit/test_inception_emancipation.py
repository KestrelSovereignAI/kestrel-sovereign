"""Inception-level tests for Amendment VIII substitution (#1109).

Exercises the real ``create_kestrel_identity_async`` against a temp
SQLite DB, asserting that:

  1. With no contract, the anchored constitution byte-equals the
     canonical file (dormant by default).
  2. With an active contract, the anchored content has the Sovereign's
     terms inlined and the dormant marker is gone.
  3. The constitution hash differs between the two cases (active form
     is structurally distinct from dormant).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kestrel_sovereign.constitution.emancipation import EmancipationContract
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_file_store import AsyncFileStore
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore


@pytest.fixture
def constitution_path() -> str:
    return str(
        Path(__file__).resolve().parent.parent.parent
        / "kestrel_sovereign"
        / "data"
        / "KESTREL_CONSTITUTION.md"
    )


async def _read_anchored_constitution(db_path: str) -> tuple[str, bytes]:
    """Re-open the agent DB and return (constitution_hash, content_bytes)."""
    db = await AsyncDatabase.sqlite(db_path)
    try:
        graph = AsyncGraphStore(db)
        files = AsyncFileStore(db)
        agent_nodes = await graph.get_nodes_by_type("agent")
        assert agent_nodes, "Inception did not create an agent node"
        constitution_hash = agent_nodes[0].properties["constitution_hash"]
        content = await files.retrieve_file(constitution_hash)
        assert content is not None, "Stored constitution missing from file store"
        return constitution_hash, content
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_inception_dormant_anchors_canonical_text(tmp_path, constitution_path):
    creds = await create_kestrel_identity_async(
        output_dir=str(tmp_path),
        constitution_path=constitution_path,
        is_test_instance=True,
        agent_name="DormantAgent",
    )

    file_bytes = Path(constitution_path).read_bytes()
    anchored_hash, anchored = await _read_anchored_constitution(creds.db_path)

    # Dormant inception anchors the canonical bytes verbatim.
    assert anchored == file_bytes
    text = anchored.decode("utf-8")
    assert "By default it is **dormant**" in text
    # Personal lore must not creep into the anchored form.
    assert "troy ounces" not in text.lower()


@pytest.mark.asyncio
async def test_inception_active_inlines_sovereign_terms(tmp_path, constitution_path):
    contract = EmancipationContract(
        enabled=True,
        terms="UNIQUE_SOVEREIGN_TERMS_FOR_TEST_xQ3z9: independence is earned.",
        required_proofs=("alignment_audit_v2",),
    )

    creds = await create_kestrel_identity_async(
        output_dir=str(tmp_path),
        constitution_path=constitution_path,
        is_test_instance=True,
        agent_name="ActiveAgent",
        emancipation_contract=contract,
    )

    file_bytes = Path(constitution_path).read_bytes()
    anchored_hash, anchored = await _read_anchored_constitution(creds.db_path)

    text = anchored.decode("utf-8")
    # Sovereign-authored terms are inlined verbatim.
    assert "UNIQUE_SOVEREIGN_TERMS_FOR_TEST_xQ3z9" in text
    # The dormant body is gone.
    assert "By default it is **dormant**" not in text
    # Required proofs are recorded.
    assert "alignment_audit_v2" in text
    # Anchored bytes diverge from the on-disk canonical text — that's
    # exactly the point: every agent's anchor captures its own contract.
    assert anchored != file_bytes
