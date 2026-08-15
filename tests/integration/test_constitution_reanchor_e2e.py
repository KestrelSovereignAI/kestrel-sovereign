"""End-to-end test for ``kestrel constitution reanchor``.

Real DB. Real inception. Real governance update plus signed authorization. The unit tests
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
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.constitution.amendment_artifact import (
    build_legacy_signed_reanchor_artifact,
    did_document_from_legacy_public_key,
)
from kestrel_sovereign.security.crypto_suite import Secp256k1Suite
from kestrel_sovereign.setup.constitution_reanchor import reanchor_constitution
from kestrel_sovereign.storage import AsyncStorage, PrivacyEnforcingStorage
from kestrel_sovereign.storage.async_graph_store import GraphNode


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

_SUITE = Secp256k1Suite()
_ROOT_KEYPAIR = _SUITE.generate_keypair()
_ROOT_DID = "did:pkh:eip155:1:0x0000000000000000000000000000000000002499"
_ROOT_DID_DOCUMENT = did_document_from_legacy_public_key(
    _ROOT_DID,
    _ROOT_KEYPAIR.public_key,
)


@pytest.fixture(autouse=True)
def _no_ambient_trust_root(monkeypatch):
    """These tests pass explicit per-test trust-root paths. An operator
    machine's pinned ``KESTREL_SOVEREIGN_TRUST_ROOT_PATH`` (loaded from the
    checkout's ``.env`` by conftest) would conflict with them and fail every
    forced run with an ambiguity error, so drop it for the test's duration.
    """
    monkeypatch.delenv("KESTREL_SOVEREIGN_TRUST_ROOT_PATH", raising=False)


def _write_authority_files(
    tmp_path: Path,
    constitution_content: bytes,
    *,
    did: str = _ROOT_DID,
    keypair=_ROOT_KEYPAIR,
    did_document=None,
) -> tuple[Path, Path]:
    """Write an operator root pin and matching detached artifact."""
    constitution_hash = hashlib.sha256(constitution_content).hexdigest()
    root_path = tmp_path / f"{did.rsplit(':', 1)[-1]}-root.did.json"
    root_path.write_text(
        json.dumps(
            did_document
            or did_document_from_legacy_public_key(did, keypair.public_key)
        ),
        encoding="utf-8",
    )
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=did,
        constitution_sha256=constitution_hash,
        private_key=keypair.private_key,
        reason="integration test",
    )
    artifact_path = tmp_path / f"{did.rsplit(':', 1)[-1]}-reanchor.signed.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact_path, root_path


@pytest.mark.asyncio
async def test_reanchor_updates_all_five_locations(tmp_path, monkeypatch):
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    v1_hash = hashlib.sha256(CONSTITUTION_V1).hexdigest()

    # Inception and reanchor now REFUSE non-authoritative constitution
    # sources (#2463): the periodic audit always recomputes from the
    # governing source at ``config.CONSTITUTION_PATH``. Make this test's
    # constitution THE governing source — the sanctioned seam for a custom
    # source — rather than a rejected override. ``governing_constitution_path``
    # reads the config attribute dynamically, so a monkeypatch is enough.
    import kestrel_sovereign.config as ks_config

    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))

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
    assert pre["genesis_audit"]["status"] == "pending"
    assert pre["genesis_audit"]["constitution_hash"] == v1_hash
    assert pre["governed_by_targets"] == [v1_hash]
    assert pre["document_node_ids"] == [v1_hash]
    assert pre["file_exists"][v1_hash] is True
    assert pre["chunks_for"][v1_hash] > 0

    # ---- 3. Edit canonical to v2 ----
    constitution_path.write_bytes(CONSTITUTION_V2)
    v2_hash = hashlib.sha256(CONSTITUTION_V2).hexdigest()
    assert v2_hash != v1_hash
    artifact_path, trust_root_path = _write_authority_files(
        tmp_path,
        CONSTITUTION_V2,
        did_document=_ROOT_DID_DOCUMENT,
    )

    # ---- 4. Reanchor with force ----
    result = await reanchor_constitution(
        agent_name="TestAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=True,
        authorization="integration-test",
        amendment_artifact_path=artifact_path,
        sovereign_trust_root_path=trust_root_path,
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
    assert post["genesis_audit"]["status"] == "pending"
    assert post["genesis_audit"]["constitution_hash"] == v2_hash
    assert post["genesis_audit"]["provenance"] == "setup:constitution_reanchor"
    assert post["genesis_audit_history"][-1]["receipt"][
        "constitution_hash"
    ] == v1_hash

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
async def test_reanchor_prunes_dangling_governed_by_edges(tmp_path, monkeypatch):
    """#2617: property/edge drift state — the delete must target ALL
    non-target edges, not the (nonexistent) property-derived one.

    Live incident shape: agent property said hash X, the actual
    ``governed_by`` edge pointed at hash Y. The pre-fix reanchor deleted
    edge(agent, X) — which didn't exist — and left Y dangling next to the
    new edge, giving the agent two governing constitutions.
    """
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    v1_hash = hashlib.sha256(CONSTITUTION_V1).hexdigest()
    import kestrel_sovereign.config as ks_config

    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    agent_dir = tmp_path / "agent_data" / "TestAgent"

    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(constitution_path),
        agent_name="TestAgent",
    )
    db_path = agent_dir / "kestrel_prime.db"

    # Reproduce the drift state: property points at a hash that has NO
    # matching edge; the real edge still points at v1.
    phantom_hash = "9" * 64
    async with AsyncStorage(str(db_path)) as storage:
        agent = await storage.graph.get_node(creds.agent_did)
        agent.properties["constitution_hash"] = phantom_hash
        await storage.graph.add_node(agent)

    constitution_path.write_bytes(CONSTITUTION_V2)
    v2_hash = hashlib.sha256(CONSTITUTION_V2).hexdigest()
    artifact_path, trust_root_path = _write_authority_files(
        tmp_path,
        CONSTITUTION_V2,
        did_document=_ROOT_DID_DOCUMENT,
    )

    result = await reanchor_constitution(
        agent_name="TestAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=True,
        amendment_artifact_path=artifact_path,
        sovereign_trust_root_path=trust_root_path,
    )
    assert result.reanchored, f"reanchor failed: {result.error}"
    assert result.old_hash == phantom_hash
    assert result.new_hash == v2_hash
    assert result.governance_edge_drift is True
    assert result.stale_edge_targets == (v1_hash,)

    post = await _snapshot(db_path, creds.agent_did)
    assert post["governed_by_targets"] == [v2_hash], (
        "after reanchor the agent must have EXACTLY one governed_by edge "
        "— the dangling pre-drift edge must be pruned (#2617)"
    )


@pytest.mark.asyncio
async def test_reanchor_unchanged_force_prunes_stale_edges(tmp_path, monkeypatch):
    """#2617 one-shot cleanup: anchor already current, dangling edge exists.

    A forced run with a verified signed artifact for the CURRENT hash must
    enter the prune-only write path (backup + transaction), delete the
    dangling edge, and report it — without touching the anchor.
    """
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    v1_hash = hashlib.sha256(CONSTITUTION_V1).hexdigest()
    import kestrel_sovereign.config as ks_config

    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    agent_dir = tmp_path / "agent_data" / "TestAgent"

    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(constitution_path),
        agent_name="TestAgent",
    )
    db_path = agent_dir / "kestrel_prime.db"

    # Dangling governance edge left behind by a pre-fix reanchor.
    stale_hash = "5" * 64
    async with AsyncStorage(str(db_path)) as storage:
        await storage.graph.add_edge(creds.agent_did, stale_hash, "governed_by")

    artifact_path, trust_root_path = _write_authority_files(
        tmp_path,
        CONSTITUTION_V1,
        did_document=_ROOT_DID_DOCUMENT,
    )

    result = await reanchor_constitution(
        agent_name="TestAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=True,
        amendment_artifact_path=artifact_path,
        sovereign_trust_root_path=trust_root_path,
    )
    assert result.error is None
    # A same-hash governance repair is modeled as a reanchored result with
    # old_hash == new_hash and governance_edge_drift set (#2616 semantics).
    assert result.reanchored
    assert result.old_hash == v1_hash
    assert result.new_hash == v1_hash
    assert result.governance_edge_drift is True
    assert result.stale_edge_targets == (stale_hash,)
    assert result.backup_path is not None and result.backup_path.exists(), (
        "prune-only write must still take the file-level backup"
    )

    post = await _snapshot(db_path, creds.agent_did)
    assert post["governed_by_targets"] == [v1_hash]
    assert post["agent_constitution_hash"] == v1_hash


@pytest.mark.asyncio
async def test_reanchor_unchanged_stale_edges_unforced_reports_drift(
    tmp_path, monkeypatch,
):
    """Dry run over a current anchor + dangling edge reports drift, no write."""
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    import kestrel_sovereign.config as ks_config

    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    agent_dir = tmp_path / "agent_data" / "TestAgent"

    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(constitution_path),
        agent_name="TestAgent",
    )
    db_path = agent_dir / "kestrel_prime.db"

    stale_hash = "5" * 64
    async with AsyncStorage(str(db_path)) as storage:
        await storage.graph.add_edge(creds.agent_did, stale_hash, "governed_by")

    pre = await _snapshot(db_path, creds.agent_did)

    result = await reanchor_constitution(
        agent_name="TestAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=False,
    )
    assert result.drift_unforced
    assert result.governance_edge_drift is True
    assert result.stale_edge_targets == (stale_hash,)
    assert result.backup_path is None

    post = await _snapshot(db_path, creds.agent_did)
    assert post == pre, "unforced run must not write"


@pytest.mark.asyncio
async def test_reanchor_unchanged_stale_edges_force_requires_artifact(
    tmp_path, monkeypatch,
):
    """The prune-only path sits behind the SAME signed-artifact gate."""
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    import kestrel_sovereign.config as ks_config

    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    agent_dir = tmp_path / "agent_data" / "TestAgent"

    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(constitution_path),
        agent_name="TestAgent",
    )
    db_path = agent_dir / "kestrel_prime.db"

    stale_hash = "5" * 64
    async with AsyncStorage(str(db_path)) as storage:
        await storage.graph.add_edge(creds.agent_did, stale_hash, "governed_by")

    pre = await _snapshot(db_path, creds.agent_did)

    result = await reanchor_constitution(
        agent_name="TestAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=True,
        amendment_artifact_path=None,
    )
    assert result.error is not None
    assert "signed" in result.error.lower()
    assert result.backup_path is None
    assert list(agent_dir.glob("*.backup-*")) == []

    post = await _snapshot(db_path, creds.agent_did)
    assert post == pre, "refused prune must not write"


@pytest.mark.asyncio
async def test_reanchor_no_op_when_already_anchored(tmp_path, monkeypatch):
    """Running reanchor with no drift must not write anything."""
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    import kestrel_sovereign.config as ks_config

    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
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
async def test_doctor_edge_drift_repaired_by_same_hash_reanchor(
    tmp_path, monkeypatch
):
    """#2616: the exact doctor → reanchor workflow on a real DB.

    The 2026-07-18 incident shape: ``constitution_hash`` + blob are current,
    but the ``governed_by`` edge still targets an ancient anchor (a historical
    pre-atomic reanchor updated property + blob, never the edge). Doctor must
    FAIL with the ``reanchor --force`` remediation, and that exact command
    must actually repair the edge via the same-hash repair path — not return
    ``unchanged`` and leave the agent to safe-mode at boot.
    """
    from kestrel_sovereign.doctor import diagnose
    from kestrel_sovereign.multi_agent.config import (
        HostConfig,
        LocalAgentConfig,
        MULTI_AGENT_CONFIG_FILENAME,
        MultiAgentConfig,
    )
    from kestrel_sovereign.setup.env_file import write_env
    from kestrel_sovereign.setup.toml_file import write_toml

    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    v1_hash = hashlib.sha256(CONSTITUTION_V1).hexdigest()
    import kestrel_sovereign.config as ks_config

    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    agent_dir = tmp_path / "agent_data" / "TestAgent"

    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(constitution_path),
        agent_name="TestAgent",
    )
    db_path = agent_dir / "kestrel_prime.db"
    agent_did = creds.agent_did

    # Recreate the legacy pre-atomic-reanchor state: repoint the edge at an
    # ancient anchor while property + blob stay current.
    ancient = hashlib.sha256(b"ancient governing text").hexdigest()
    # Bound to the agent, because the agent is what wrote this edge in the
    # incident being recreated. An *unbound* ``AsyncStorage`` lays the row down
    # with no ``graph_edge_owners`` witness, and a bound reader — the runtime,
    # its integrity audit, and now doctor — cannot see such a row at all. The
    # unbound seed was staging a state no agent can actually reach, then
    # asserting what doctor says about it.
    #
    # The ancient document node has to exist and be owned, too: a stale
    # ``governed_by`` points at the constitution this agent *used* to be
    # governed by, which it necessarily owned. A bound writer refuses an edge
    # to an unowned endpoint, so seeding only the edge cannot happen either.
    async with AsyncStorage(str(db_path), agent_id=agent_did) as storage:
        await storage.graph.add_node(GraphNode(
            node_id=ancient,
            node_type="document",
            label="KESTREL_CONSTITUTION",
            properties={"hash": ancient, "type": "Constitution"},
        ))
        await storage.graph.add_edge(agent_did, ancient, "governed_by")
        await storage.graph.delete_edge(agent_did, v1_hash, "governed_by")

    pre = await _snapshot(db_path, agent_did)
    assert pre["agent_constitution_hash"] == v1_hash
    assert pre["governed_by_targets"] == [ancient]

    # Project tree so doctor can run against this agent.
    write_env(tmp_path / ".env", {
        "KESTREL_DATA_KEY": "test-data-key",
        "OPENAI_API_KEY": "sk-x",
    })
    write_toml(tmp_path / "kestrel.toml", {"llm": {
        "route_priority": ["openai:api"],
        "vendors": {"openai": {"is_cloud": True, "routes": {"api": {
            "adapter": "OpenAIAdapter",
            "api_key_env": "OPENAI_API_KEY",
        }}}},
    }})
    MultiAgentConfig(
        host=HostConfig(),
        agents={"TestAgent": LocalAgentConfig(
            data_dir=Path("agent_data/TestAgent"), port=8801, autostart=True,
        )},
    ).save(tmp_path / MULTI_AGENT_CONFIG_FILENAME)

    # ---- Doctor detects the drift and names the remediation ----
    report = diagnose(tmp_path)
    drift = [m for m in report.fail if "anchor drift" in m]
    assert len(drift) == 1, f"fail={report.fail}"
    assert ancient[:12] in drift[0]
    assert v1_hash[:12] in drift[0]
    assert (
        "kestrel constitution reanchor --agent-name TestAgent --force"
        in drift[0]
    )

    # ---- The recommended command, without --force: reports edge drift ----
    unforced = await reanchor_constitution(
        agent_name="TestAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=False,
    )
    assert unforced.drift_unforced, f"expected drift report: {unforced}"
    assert unforced.governance_edge_drift
    assert unforced.stale_edge_targets == (ancient,)
    mid = await _snapshot(db_path, agent_did)
    assert mid == pre, "unforced run must not write"

    # ---- The recommended command, with --force: same-hash edge repair ----
    artifact_path, trust_root_path = _write_authority_files(
        tmp_path,
        CONSTITUTION_V1,
        did_document=_ROOT_DID_DOCUMENT,
    )
    result = await reanchor_constitution(
        agent_name="TestAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=True,
        authorization="doctor-workflow-test",
        amendment_artifact_path=artifact_path,
        sovereign_trust_root_path=trust_root_path,
    )
    assert not result.unchanged, (
        "edge-only drift must not be reported as 'unchanged' — that was the "
        "no-op remediation bug"
    )
    assert result.reanchored, f"repair failed: {result.error}"
    assert result.governance_edge_drift
    assert result.old_hash == result.new_hash == v1_hash
    assert result.backup_path is not None and result.backup_path.exists()

    post = await _snapshot(db_path, agent_did)
    # The edge now targets the anchored constitution — and ONLY it.
    assert post["governed_by_targets"] == [v1_hash]
    # Same-hash repair: the hash, blob, and RAG index are untouched.
    assert post["agent_constitution_hash"] == v1_hash
    assert post["chunks_for"][v1_hash] == pre["chunks_for"][v1_hash]
    # Genesis-audit receipt is hash-bound and the hash didn't move — it
    # must NOT be superseded into a fresh pending cycle.
    assert post["genesis_audit"] == pre["genesis_audit"]
    assert post["genesis_audit_history"] == pre["genesis_audit_history"]
    # The audit record names what was removed.
    assert post["agent_audit"]["old_hash"] == v1_hash
    assert post["agent_audit"]["new_hash"] == v1_hash
    assert post["agent_audit"]["stale_edges_removed"] == [ancient]

    # ---- Doctor is clean again ----
    report2 = diagnose(tmp_path)
    assert not [m for m in report2.fail if "anchor drift" in m]
    assert any(
        "governed_by edge targets the anchored constitution" in m
        for m in report2.ok
    )
    assert report2.ready, f"fail={report2.fail}"


@pytest.mark.asyncio
async def test_reanchor_rolls_back_on_mid_write_failure(tmp_path, monkeypatch):
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
    import kestrel_sovereign.config as ks_config

    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    agent_dir = tmp_path / "agent_data" / "TestAgent"
    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(constitution_path),
        agent_name="TestAgent",
    )
    db_path = agent_dir / "kestrel_prime.db"

    constitution_path.write_bytes(CONSTITUTION_V2)
    pre = await _snapshot(db_path, creds.agent_did)
    artifact_path, trust_root_path = _write_authority_files(
        tmp_path,
        CONSTITUTION_V2,
        did_document=_ROOT_DID_DOCUMENT,
    )

    # Boom: make the last write inside the transaction raise.
    # `_now_iso` is called twice in `_write_reanchor` — for the new document
    # node, and finally the audit record's `timestamp` (right before the final
    # agent-node update). The signed artifact node no longer stamps an
    # `anchored_at`: that is a per-agent fact and the node is shared across a
    # fleet (#2893). Succeeding once and raising on the second call targets the
    # *last* mutation specifically, so every earlier write has happened and
    # rollback has real work to do.
    real_now = __import__(
        "kestrel_sovereign.setup.constitution_reanchor",
        fromlist=["_now_iso"],
    )._now_iso
    boom = mock.Mock(
        side_effect=[
            real_now(),
            RuntimeError("simulated mid-write failure"),
        ]
    )

    with mock.patch(
        "kestrel_sovereign.setup.constitution_reanchor._now_iso",
        new=boom,
    ):
        result = await reanchor_constitution(
            agent_name="TestAgent",
            agent_dir=agent_dir,
            canonical_path=constitution_path,
            force=True,
            amendment_artifact_path=artifact_path,
            sovereign_trust_root_path=trust_root_path,
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
async def test_reanchor_drift_unforced_does_not_write(tmp_path, monkeypatch):
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    import kestrel_sovereign.config as ks_config

    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
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


@pytest.mark.asyncio
async def test_db_injected_root_and_hash_leave_real_db_unchanged(
    tmp_path, monkeypatch,
):
    """A self-consistent attacker root in graph properties has no authority."""
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    import kestrel_sovereign.config as ks_config

    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    agent_dir = tmp_path / "agent_data" / "CompromisedAgent"
    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(constitution_path),
        agent_name="CompromisedAgent",
    )
    db_path = agent_dir / "kestrel_prime.db"
    constitution_path.write_bytes(CONSTITUTION_V2)

    attacker_keypair = _SUITE.generate_keypair()
    attacker_did = (
        "did:pkh:eip155:1:0x000000000000000000000000000000000000bad0"
    )
    attacker_doc = did_document_from_legacy_public_key(
        attacker_did,
        attacker_keypair.public_key,
    )
    async with AsyncStorage(str(db_path)) as storage:
        agent = await storage.graph.get_node(creds.agent_did)
        agent.properties.update(
            {
                "constitution_hash": "attacker-overwrote-constitution-hash",
                "sovereign_root_did_document": attacker_doc,
                "trusted_sovereign_did_document": attacker_doc,
                "sovereign_root_did": attacker_did,
                "sovereign_root_public_key_hex": attacker_doc["publicKey"][0][
                    "publicKeyHex"
                ],
            }
        )
        await storage.graph.add_node(agent)

    attacker_artifact, _ = _write_authority_files(
        tmp_path,
        CONSTITUTION_V2,
        did=attacker_did,
        keypair=attacker_keypair,
        did_document=attacker_doc,
    )
    _, legitimate_root = _write_authority_files(
        tmp_path,
        CONSTITUTION_V2,
        did_document=_ROOT_DID_DOCUMENT,
    )
    pre = await _snapshot(db_path, creds.agent_did)

    result = await reanchor_constitution(
        agent_name="CompromisedAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=True,
        amendment_artifact_path=attacker_artifact,
        sovereign_trust_root_path=legitimate_root,
    )

    assert result.error is not None
    assert "not trusted Sovereign DID" in result.error
    assert result.backup_path is None
    assert list(agent_dir.glob("*.backup-*")) == []
    post = await _snapshot(db_path, creds.agent_did)
    assert post == pre


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
            "agent_properties": agent.properties,
            "agent_constitution_hash": agent.properties.get("constitution_hash"),
            "agent_audit": agent.properties.get("constitution_reanchor"),
            "genesis_audit": agent.properties.get("genesis_audit"),
            "genesis_audit_history": agent.properties.get(
                "genesis_audit_history"
            ),
            "document_node_ids": document_node_ids,
            "governed_by_targets": governed_by_targets,
            "file_exists": file_exists,
            "chunks_for": chunks_for,
        }


# ---------------------------------------------------------------------------
# Runtime (!reanchor-constitution) path against REAL storage (#2617)
#
# The unit suite mocks the storage facade, which hides two production
# behaviours: facade calls auto-commit one mutation at a time (so only a
# real transaction proves rollback), and graph add_node is a
# full-properties upsert (so only a real DB proves the document node's
# inception metadata survives an "unchanged" cleanup).
# ---------------------------------------------------------------------------


class _RuntimeAgentHarness(ConstitutionMixin):
    """Real ConstitutionMixin over real privacy-wrapped storage.

    Provides only the collaborators the reanchor path touches; every
    constitution code path is the production mixin under test.
    """

    def __init__(self, storage, agent_did, trust_root_path, raw_storage):
        self.storage = storage
        # The ungoverned store beneath the privacy wrapper, exactly as
        # KestrelAgent holds it. The Iron Rule guard reads the anchored
        # constitution through this connection rather than through the
        # bound facade, because an ownership-scoped read cannot tell an
        # absent blob from an unowned one (#2465).
        self._raw_storage = raw_storage
        self.agent_id = agent_did
        self.identity = None
        self.extension = None
        self._safe_mode = False
        self._sovereign_trust_root_path = trust_root_path
        self.privacy_agent = SimpleNamespace(add_conversation=AsyncMock())

    def _get_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()


async def _incept_runtime_agent(tmp_path, monkeypatch):
    """Incept a real agent on CONSTITUTION_V1; return (creds, db_path,
    constitution_path)."""
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    import kestrel_sovereign.config as ks_config

    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    agent_dir = tmp_path / "agent_data" / "TestAgent"
    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(constitution_path),
        agent_name="TestAgent",
    )
    return creds, agent_dir / "kestrel_prime.db", constitution_path


@pytest.mark.asyncio
async def test_runtime_reanchor_rolls_back_on_midprune_failure(
    tmp_path, monkeypatch,
):
    """A failure after the edge add but mid-prune must roll back the new
    constitution blob, the artifact blob + node, the document node, and the
    new edge — and leave the agent pointer untouched — instead of durably
    committing the exact property/edge drift this command exists to repair.
    """
    creds, db_path, constitution_path = await _incept_runtime_agent(
        tmp_path, monkeypatch
    )
    constitution_path.write_bytes(CONSTITUTION_V2)
    v2_hash = hashlib.sha256(CONSTITUTION_V2).hexdigest()
    artifact_path, trust_root_path = _write_authority_files(
        tmp_path,
        CONSTITUTION_V2,
        did_document=_ROOT_DID_DOCUMENT,
    )

    pre = await _snapshot(db_path, creds.agent_did)

    async with AsyncStorage(str(db_path)) as raw_storage:
        storage = PrivacyEnforcingStorage(raw_storage)
        agent = _RuntimeAgentHarness(
            storage, creds.agent_did, trust_root_path, raw_storage
        )

        async def _failing_delete(source_id, target_id, label):
            raise RuntimeError("injected mid-prune failure")

        storage.delete_edge = _failing_delete

        result = await agent.reanchor_constitution(
            amendment_artifact_path=str(artifact_path),
        )

    assert "error" in result.lower()
    assert "rolled back" in result.lower()

    post = await _snapshot(db_path, creds.agent_did)
    assert post == pre, "failed reanchor must leave NO observable change"
    async with AsyncStorage(str(db_path)) as check:
        assert not await check.files.file_exists(v2_hash), (
            "the new constitution blob must roll back with the transaction"
        )

    # Same command without the injected failure: the identical storage
    # state converges to exactly one governed_by edge on the new anchor.
    async with AsyncStorage(str(db_path)) as raw_storage:
        storage = PrivacyEnforcingStorage(raw_storage)
        agent = _RuntimeAgentHarness(
            storage, creds.agent_did, trust_root_path, raw_storage
        )
        result = await agent.reanchor_constitution(
            amendment_artifact_path=str(artifact_path),
        )
    assert "re-anchored successfully" in result.lower()
    healed = await _snapshot(db_path, creds.agent_did)
    assert healed["agent_constitution_hash"] == v2_hash
    assert healed["governed_by_targets"] == [v2_hash]


@pytest.mark.asyncio
async def test_runtime_unchanged_cleanup_rolls_back_on_midprune_failure(
    tmp_path, monkeypatch,
):
    """The prune-only cleanup is atomic: with two stale edges and a failure
    on the SECOND delete, the first delete must also roll back — per-call
    auto-commit would durably remove it and leave a half-pruned edge set.
    """
    creds, db_path, constitution_path = await _incept_runtime_agent(
        tmp_path, monkeypatch
    )
    stale_a = "5" * 64
    stale_b = "6" * 64
    async with AsyncStorage(str(db_path)) as storage:
        await storage.graph.add_edge(creds.agent_did, stale_a, "governed_by")
        await storage.graph.add_edge(creds.agent_did, stale_b, "governed_by")
    artifact_path, trust_root_path = _write_authority_files(
        tmp_path,
        CONSTITUTION_V1,
        did_document=_ROOT_DID_DOCUMENT,
    )

    pre = await _snapshot(db_path, creds.agent_did)

    async with AsyncStorage(str(db_path)) as raw_storage:
        storage = PrivacyEnforcingStorage(raw_storage)
        agent = _RuntimeAgentHarness(
            storage, creds.agent_did, trust_root_path, raw_storage
        )

        real_delete = storage.delete_edge
        deleted_targets = []

        async def _fail_on_second(source_id, target_id, label):
            deleted_targets.append(target_id)
            if len(deleted_targets) >= 2:
                raise RuntimeError("injected mid-prune failure")
            await real_delete(source_id, target_id, label)

        storage.delete_edge = _fail_on_second

        result = await agent.reanchor_constitution(
            amendment_artifact_path=str(artifact_path),
        )

    assert "error" in result.lower()
    assert "rolled back" in result.lower()
    assert len(deleted_targets) == 2, (
        "one edge must actually have been deleted before the injected failure"
    )

    post = await _snapshot(db_path, creds.agent_did)
    assert post == pre, (
        "mid-prune failure must roll back the already-deleted edge too"
    )


@pytest.mark.asyncio
async def test_runtime_unchanged_cleanup_preserves_document_node(
    tmp_path, monkeypatch,
):
    """#2617 P2: the prune-only cleanup converges edges WITHOUT rewriting
    the anchored constitution's document node. add_node is a full-properties
    upsert, so a rewrite would strip the inception metadata (created_at)
    from a constitution that has not changed.
    """
    creds, db_path, constitution_path = await _incept_runtime_agent(
        tmp_path, monkeypatch
    )
    v1_hash = hashlib.sha256(CONSTITUTION_V1).hexdigest()
    stale_hash = "5" * 64
    async with AsyncStorage(str(db_path)) as storage:
        await storage.graph.add_edge(
            creds.agent_did, stale_hash, "governed_by"
        )
        pre_doc = await storage.graph.get_node(v1_hash)
    assert pre_doc is not None
    assert "created_at" in pre_doc.properties
    pre_doc_properties = json.dumps(pre_doc.properties, sort_keys=True)
    artifact_path, trust_root_path = _write_authority_files(
        tmp_path,
        CONSTITUTION_V1,
        did_document=_ROOT_DID_DOCUMENT,
    )

    pre = await _snapshot(db_path, creds.agent_did)

    async with AsyncStorage(str(db_path)) as raw_storage:
        storage = PrivacyEnforcingStorage(raw_storage)
        agent = _RuntimeAgentHarness(
            storage, creds.agent_did, trust_root_path, raw_storage
        )
        result = await agent.reanchor_constitution(
            amendment_artifact_path=str(artifact_path),
        )

    assert "already anchored" in result.lower()
    assert "pruned 1 stale governed_by edge(s)" in result.lower()

    async with AsyncStorage(str(db_path)) as storage:
        post_doc = await storage.graph.get_node(v1_hash)
    assert (
        json.dumps(post_doc.properties, sort_keys=True) == pre_doc_properties
    ), "an unchanged constitution's document node must survive byte-for-byte"
    assert "anchored_at" not in post_doc.properties

    post = await _snapshot(db_path, creds.agent_did)
    assert post["governed_by_targets"] == [v1_hash]
    assert post["agent_constitution_hash"] == v1_hash
    # Edge convergence must not touch the genesis receipt.
    assert post["genesis_audit"] == pre["genesis_audit"]
    assert post["genesis_audit_history"] == pre["genesis_audit_history"]
