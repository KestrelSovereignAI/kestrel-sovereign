"""#2890 — ``kestrel constitution reanchor`` against a PostgreSQL runtime.

The defect this pins cannot be reproduced on SQLite, because on SQLite the
local ``kestrel_prime.db`` *is* the database the runtime reads. On a host
configured with ``KESTREL_DB_BACKEND=postgres`` they are two different
databases: the anchor holds the birth record (#2871) and the agent's
governance lives in PostgreSQL. Resolving the anchor unconditionally produced
a reanchor that reported success, took a timestamped backup, and changed
nothing the running agent is governed by.

Run against any throwaway PostgreSQL:

    TEST_POSTGRES_URL=postgresql://u:p@127.0.0.1:5432/db pytest \
        tests/integration/test_constitution_reanchor_postgres.py

Skipped when that is not set, so CI (which has no PostgreSQL) stays green.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from kestrel_sovereign.constitution.amendment_artifact import (
    build_legacy_signed_reanchor_artifact,
    did_document_from_legacy_public_key,
)
from kestrel_sovereign.security.crypto_suite import Secp256k1Suite
from kestrel_sovereign.setup.constitution_reanchor import reanchor_constitution
from kestrel_sovereign.storage import AsyncStorage, GraphNode

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

POSTGRES_URL = (
    os.environ.get("TEST_POSTGRES_URL")
    or os.environ.get("KESTREL_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
)

if not POSTGRES_URL:  # pragma: no cover - environment gate
    pytest.skip(
        "TEST_POSTGRES_URL / KESTREL_DATABASE_URL / DATABASE_URL required",
        allow_module_level=True,
    )

CONSTITUTION_V1 = b"""# Kestrel Constitution (PG target test, v1)

## Book I: Universal Values

Honesty. Sovereignty. Transparency.
""" * 6

CONSTITUTION_V2 = b"""# Kestrel Constitution (PG target test, v2 - AMENDED)

## Book I: Universal Values

Honesty. Sovereignty. Transparency. Calibrated uncertainty.
""" * 6

#: A THIRD constitution, for the neighbouring tenant. Seeding the neighbour
#: with the same document as the subject makes a cross-tenant read
#: indistinguishable from a correct one — their ``governed_by`` targets are
#: then the same hash, so dropping ``WHERE source_id = ?`` changes nothing an
#: assertion can see. Mutation testing found exactly that.
CONSTITUTION_NEIGHBOUR = b"""# Kestrel Constitution (PG target test, neighbour)

## Book I: Universal Values

Honesty. Sovereignty. Transparency. A different governing document.
""" * 6

_SUITE = Secp256k1Suite()
_ROOT_KEYPAIR = _SUITE.generate_keypair()
_ROOT_DID = "did:pkh:eip155:1:0x0000000000000000000000000000000000002890"
AGENT_DID = "did:pkh:eip155:1:0x000000000000000000000000000000000000a890"
OTHER_DID = "did:pkh:eip155:1:0x000000000000000000000000000000000000b890"


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch):
    """The operator machine must not decide this test's answers.

    ``KESTREL_SOVEREIGN_TRUST_ROOT_PATH`` would conflict with the explicit
    per-test root; ``KESTREL_DB_BACKEND`` would make the "explicit arguments
    are what chose PostgreSQL" claim untestable.
    """
    monkeypatch.delenv("KESTREL_SOVEREIGN_TRUST_ROOT_PATH", raising=False)
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)


@pytest.fixture
async def pg():
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.postgres(POSTGRES_URL)
    try:
        for did in (AGENT_DID, OTHER_DID):
            await _purge_agent(db, did)
        yield db
    finally:
        for did in (AGENT_DID, OTHER_DID):
            await _purge_agent(db, did)
        await db.close()


async def _purge_agent(db, agent_did: str = AGENT_DID) -> None:
    """Remove every row this module's agent owns, shared rows included.

    Content-addressed ``files`` / ``graph_nodes`` rows are shared across
    tenants. Deleting only the owner rows leaves them ownerless, and the next
    run's ``store_file`` then raises "Cannot claim an unowned legacy file" —
    the suite would fail in setup having itself created the shape that bricks
    a real agent (learned in #2871).
    """
    owned_files = [
        row[0]
        for row in await db.fetchall(
            "SELECT content_hash FROM file_owners WHERE agent_id = $1", (agent_did,)
        )
    ]
    owned_nodes = [
        row[0]
        for row in await db.fetchall(
            "SELECT node_id FROM graph_node_owners WHERE agent_id = $1", (agent_did,)
        )
    ]
    await db.execute_commit(
        "DELETE FROM graph_edge_owners WHERE agent_id = $1", (agent_did,)
    )
    await db.execute_commit(
        "DELETE FROM graph_edges WHERE source_id = $1", (agent_did,)
    )
    for content_hash in owned_files:
        await db.execute_commit(
            "DELETE FROM document_chunks WHERE file_hash = $1", (content_hash,)
        )
        await db.execute_commit(
            "DELETE FROM file_owners WHERE content_hash = $1", (content_hash,)
        )
        await db.execute_commit(
            "DELETE FROM files WHERE content_hash = $1", (content_hash,)
        )
    for node_id in owned_nodes:
        await db.execute_commit(
            "DELETE FROM graph_node_owners WHERE node_id = $1", (node_id,)
        )
        await db.execute_commit(
            "DELETE FROM graph_nodes WHERE node_id = $1", (node_id,)
        )


def _write_authority_files(tmp_path: Path, content: bytes) -> tuple[Path, Path]:
    root_path = tmp_path / "sovereign-root.did.json"
    root_path.write_text(
        json.dumps(
            did_document_from_legacy_public_key(_ROOT_DID, _ROOT_KEYPAIR.public_key)
        ),
        encoding="utf-8",
    )
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=_ROOT_DID,
        constitution_sha256=hashlib.sha256(content).hexdigest(),
        private_key=_ROOT_KEYPAIR.private_key,
        reason="#2890 integration",
    )
    artifact_path = tmp_path / "reanchor.signed.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact_path, root_path


async def _seed_runtime_agent(constitution: bytes, agent_did: str = AGENT_DID) -> str:
    """Put a governed agent into PostgreSQL — the state boot would leave."""
    constitution_hash = hashlib.sha256(constitution).hexdigest()
    async with AsyncStorage(backend="postgres", dsn=POSTGRES_URL) as storage:
        storage.graph.bind_agent(agent_did)
        storage.files.bind_agent(agent_did)
        await storage.files.store_file(constitution, "KESTREL_CONSTITUTION.md")
        await storage.graph.add_node(
            GraphNode(
                node_id=constitution_hash,
                node_type="document",
                label="KESTREL_CONSTITUTION",
                properties={"hash": constitution_hash, "type": "Constitution"},
            )
        )
        await storage.graph.add_node(
            GraphNode(
                node_id=agent_did,
                node_type="agent",
                label="PgTargetAgent",
                properties={
                    "name": "PgTargetAgent",
                    "constitution_hash": constitution_hash,
                },
            )
        )
        await storage.graph.add_edge(agent_did, constitution_hash, "governed_by")
    return constitution_hash


async def _runtime_state(db, agent_did: str = AGENT_DID) -> tuple[str, list[str]]:
    row = await db.fetchone(
        "SELECT properties FROM graph_nodes WHERE node_id = $1", (agent_did,)
    )
    properties = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    edges = [
        r[0]
        for r in await db.fetchall(
            "SELECT target_id FROM graph_edges "
            "WHERE source_id = $1 AND label = 'governed_by'",
            (agent_did,),
        )
    ]
    return properties.get("constitution_hash"), sorted(edges)


def _make_local_anchor(agent_dir: Path, agent_did: str = AGENT_DID) -> bytes:
    """A real, non-empty ``kestrel_prime.db`` — the birth record #2871 keeps.

    Byte-equality after the run asserts the *record* is unchanged, not that
    the file was never opened. It always is: ``resolve_reanchor_target`` reads
    this agent's DID out of it through ``read_anchor_agent_did(..., INITIALIZATION)``,
    which opens ``mode=rw`` on every backend so SQLite can replay a WAL. On a
    WAL anchor that checkpoints — the bytes change while the record does not.
    This fixture is journal-mode ``delete``, so byte-equality is reachable
    here; do not read it as a promise about production anchors.
    """
    import sqlite3

    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "kestrel_prime.db"
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS graph_nodes "
            "(node_id TEXT PRIMARY KEY, node_type TEXT, label TEXT, properties TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO graph_nodes VALUES (?, 'agent', 'PgTargetAgent', '{}')",
            (agent_did,),
        )
        conn.commit()
    return path.read_bytes()


async def test_reanchor_writes_postgres_and_leaves_the_anchor_untouched(
    pg, tmp_path, monkeypatch,
):
    v1_hash = await _seed_runtime_agent(CONSTITUTION_V1)
    agent_dir = tmp_path / "agent_data" / "PgTargetAgent"
    anchor_bytes = _make_local_anchor(agent_dir, AGENT_DID)

    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V2)
    v2_hash = hashlib.sha256(CONSTITUTION_V2).hexdigest()
    import kestrel_sovereign.config as ks_config
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    artifact_path, root_path = _write_authority_files(tmp_path, CONSTITUTION_V2)

    assert await _runtime_state(pg) == (v1_hash, [v1_hash])

    result = await reanchor_constitution(
        agent_name="PgTargetAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=True,
        amendment_artifact_path=artifact_path,
        sovereign_trust_root_path=root_path,
        runtime_backend="postgres",
        runtime_dsn=POSTGRES_URL,
    )

    assert result.error is None, result.error
    assert result.reanchored is True
    assert result.target_backend == "postgres"
    assert POSTGRES_URL.rsplit("@", 1)[-1] in result.target_label

    # The governance the agent actually reads moved.
    assert await _runtime_state(pg) == (v2_hash, [v2_hash])

    # The birth record did not.
    assert (agent_dir / "kestrel_prime.db").read_bytes() == anchor_bytes


async def test_no_file_backup_is_claimed_for_a_postgres_target(
    pg, tmp_path, monkeypatch,
):
    """Naming a backup of a file the write never touches is worse than having
    none: an operator would restore it and believe they had undone the
    reanchor."""
    await _seed_runtime_agent(CONSTITUTION_V1)
    agent_dir = tmp_path / "agent_data" / "PgTargetAgent"
    _make_local_anchor(agent_dir, AGENT_DID)

    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V2)
    import kestrel_sovereign.config as ks_config
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    artifact_path, root_path = _write_authority_files(tmp_path, CONSTITUTION_V2)

    result = await reanchor_constitution(
        agent_name="PgTargetAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=True,
        amendment_artifact_path=artifact_path,
        sovereign_trust_root_path=root_path,
        runtime_backend="postgres",
        runtime_dsn=POSTGRES_URL,
    )

    assert result.error is None, result.error
    assert result.backup_path is None
    assert "no file-level backup" in (result.backup_unavailable_reason or "")
    assert not list(agent_dir.glob("*.backup-*"))


async def test_a_neighbours_agent_node_is_never_mistaken_for_this_one(
    pg, tmp_path, monkeypatch,
):
    """Deterministic probe for the tenant binding.

    Only the *neighbour* has an agent node in PostgreSQL; the agent being
    reanchored has none. A bound read asks for its own DID and finds nothing.
    An unbound one takes ``get_nodes_by_type("agent")[0]`` and can only return
    the neighbour — no row-ordering assumption required, which is what makes
    this able to observe the failure at all. The two-agent test below is the
    realistic shape; this one is the one that cannot pass by luck.
    """
    other_hash = await _seed_runtime_agent(CONSTITUTION_V1, OTHER_DID)
    agent_dir = tmp_path / "agent_data" / "PgTargetAgent"
    _make_local_anchor(agent_dir, AGENT_DID)

    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V2)
    import kestrel_sovereign.config as ks_config
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))

    result = await reanchor_constitution(
        agent_name="PgTargetAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=False,
        runtime_backend="postgres",
        runtime_dsn=POSTGRES_URL,
    )

    assert result.error is not None
    assert "no constitution_hash property" in result.error
    assert result.old_hash is None
    # The tell: an unbound read reports the neighbour's anchor as this one's.
    assert result.old_hash != other_hash
    assert other_hash not in (result.error or "")


async def test_overlay_anchor_never_writes_a_neighbours_agent_node(pg, tmp_path):
    """Same deterministic probe for ``anchor-overlay``. An unbound read here
    grants the neighbour a DANGEROUS Amendment IX overlay."""
    from kestrel_sovereign.setup.overlay_anchor import (
        OVERLAY_HASH_PROPERTY,
        anchor_overlay,
    )

    await _seed_runtime_agent(CONSTITUTION_V1, OTHER_DID)
    agent_dir = tmp_path / "agent_data" / "PgTargetAgent"
    _make_local_anchor(agent_dir, AGENT_DID)
    (agent_dir / "CONSTITUTION.md").write_bytes(b"# Overlay\n\nshell granted.\n")

    result = await anchor_overlay(
        agent_name="PgTargetAgent",
        agent_dir=agent_dir,
        runtime_backend="postgres",
        runtime_dsn=POSTGRES_URL,
    )

    assert result.error is not None
    assert "no agent identity node" in result.error
    row = await pg.fetchone(
        "SELECT properties FROM graph_nodes WHERE node_id = $1", (OTHER_DID,)
    )
    neighbour = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    assert OVERLAY_HASH_PROPERTY not in neighbour


async def test_reanchor_touches_only_the_named_agent_on_a_shared_database(
    pg, tmp_path, monkeypatch,
):
    """One PostgreSQL, two local agents — the configuration
    ``agent_manager._initialize_agent`` produces for every agent on the host.

    An unbound ``AsyncGraphStore`` scopes to ``1 = 1``, and
    ``get_nodes_by_type("agent")`` has no ORDER BY, so "the agent node" is
    whichever row the database hands back. Reanchoring one agent would then
    move the other's ``governed_by`` edge, supersede its genesis receipt, and
    stamp its ``constitution_reanchor`` record — and print the name you asked
    for.
    """
    v1_hash = await _seed_runtime_agent(CONSTITUTION_V1, AGENT_DID)
    other_v1_hash = await _seed_runtime_agent(
        CONSTITUTION_NEIGHBOUR, OTHER_DID
    )
    assert other_v1_hash != v1_hash, (
        "the neighbour must be governed by a DIFFERENT document, or a "
        "cross-tenant read is indistinguishable from a correct one"
    )
    agent_dir = tmp_path / "agent_data" / "PgTargetAgent"
    _make_local_anchor(agent_dir, AGENT_DID)

    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V2)
    v2_hash = hashlib.sha256(CONSTITUTION_V2).hexdigest()
    import kestrel_sovereign.config as ks_config
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    artifact_path, root_path = _write_authority_files(tmp_path, CONSTITUTION_V2)

    result = await reanchor_constitution(
        agent_name="PgTargetAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=True,
        amendment_artifact_path=artifact_path,
        sovereign_trust_root_path=root_path,
        runtime_backend="postgres",
        runtime_dsn=POSTGRES_URL,
    )

    assert result.error is None, result.error
    assert result.old_hash == v1_hash
    # The subject's drift was computed from ITS OWN edges: its previous
    # anchor is legitimately stale, and nothing else is. With the
    # ``source_id`` filter dropped, this read also returns the neighbour's
    # governance and reports the neighbour's constitution as a target to
    # remove — which is why the neighbour must hold a DIFFERENT document for
    # this assertion to be able to fail.
    assert result.stale_edge_targets == (v1_hash,)
    assert other_v1_hash not in result.stale_edge_targets
    assert OTHER_DID not in result.stale_edge_targets

    # The named agent moved.
    assert await _runtime_state(pg, AGENT_DID) == (v2_hash, [v2_hash])
    # The other one did not — not its hash, not its edge, not its receipt.
    assert await _runtime_state(pg, OTHER_DID) == (other_v1_hash, [other_v1_hash])
    row = await pg.fetchone(
        "SELECT properties FROM graph_nodes WHERE node_id = $1", (OTHER_DID,)
    )
    other = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    assert "constitution_reanchor" not in other
    assert "genesis_audit" not in other


async def test_overlay_anchor_touches_only_the_named_agent(pg, tmp_path):
    """The overlay hash authorizes DANGEROUS Amendment IX grants. Anchoring it
    onto a neighbouring tenant is a privilege grant to the wrong agent."""
    from kestrel_sovereign.setup.overlay_anchor import (
        OVERLAY_HASH_PROPERTY,
        anchor_overlay,
    )

    await _seed_runtime_agent(CONSTITUTION_V1, AGENT_DID)
    await _seed_runtime_agent(CONSTITUTION_V1, OTHER_DID)
    agent_dir = tmp_path / "agent_data" / "PgTargetAgent"
    _make_local_anchor(agent_dir, AGENT_DID)
    overlay = agent_dir / "CONSTITUTION.md"
    overlay.write_bytes(b"# Overlay\n\nshell_execution_host granted.\n")
    overlay_hash = hashlib.sha256(overlay.read_bytes()).hexdigest()

    result = await anchor_overlay(
        agent_name="PgTargetAgent",
        agent_dir=agent_dir,
        runtime_backend="postgres",
        runtime_dsn=POSTGRES_URL,
    )

    assert result.error is None, result.error
    for did, expected in ((AGENT_DID, overlay_hash), (OTHER_DID, None)):
        row = await pg.fetchone(
            "SELECT properties FROM graph_nodes WHERE node_id = $1", (did,)
        )
        properties = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        assert properties.get(OVERLAY_HASH_PROPERTY) == expected


async def test_one_signed_artifact_governs_a_whole_fleet(
    pg, tmp_path, monkeypatch,
):
    """The Sovereign signs one authorization; every agent under it anchors.

    A signed artifact is content-addressed, so on a shared PostgreSQL one file
    is one ``graph_nodes`` row for the fleet. It used to be the *first* agent's
    row: the node carried ``source_path``, ``anchored_at`` and ``verification``
    — all per-agent — so ``add_node`` could not admit a second owner and the
    second reanchor failed with "Cannot overwrite a graph node owned by another
    agent". #2890 turned that into a legible pre-write refusal; #2893 removes
    the need for one by putting only content-derived fields on the node.

    Both agents end up governed by v2, and both own the one artifact row.
    """
    await _seed_runtime_agent(CONSTITUTION_V1, AGENT_DID)
    await _seed_runtime_agent(CONSTITUTION_V1, OTHER_DID)
    v2_hash = hashlib.sha256(CONSTITUTION_V2).hexdigest()
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V2)
    import kestrel_sovereign.config as ks_config
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    artifact_path, root_path = _write_authority_files(tmp_path, CONSTITUTION_V2)
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    results = {}
    for did, name in ((AGENT_DID, "First"), (OTHER_DID, "Second")):
        agent_dir = tmp_path / "agent_data" / name
        _make_local_anchor(agent_dir, did)
        results[name] = await reanchor_constitution(
            agent_name=name,
            agent_dir=agent_dir,
            canonical_path=constitution_path,
            force=True,
            amendment_artifact_path=artifact_path,
            sovereign_trust_root_path=root_path,
            runtime_backend="postgres",
            runtime_dsn=POSTGRES_URL,
        )

    for name in ("First", "Second"):
        assert results[name].error is None, f"{name}: {results[name].error}"
        assert results[name].reanchored is True
    assert await _runtime_state(pg, AGENT_DID) == (v2_hash, [v2_hash])
    assert await _runtime_state(pg, OTHER_DID) == (v2_hash, [v2_hash])

    # One row, two owners — not two rows, and not one tenant's row silently
    # rewritten by the other.
    rows = await pg.fetchall(
        "SELECT properties FROM graph_nodes WHERE node_id = ?", (artifact_hash,)
    )
    assert len(rows) == 1
    owners = await pg.fetchall(
        "SELECT agent_id FROM graph_node_owners WHERE node_id = ?",
        (artifact_hash,),
    )
    assert sorted(row[0] for row in owners) == sorted([AGENT_DID, OTHER_DID])

    # And the row itself carries nothing per-agent — that is what made it
    # shareable. Every field here is fixed by the artifact bytes.
    properties = json.loads(rows[0][0]) if isinstance(rows[0][0], str) else rows[0][0]
    assert set(properties) == {
        "hash", "type", "artifact_type", "constitution_hash", "signer", "created_at",
    }
    assert "source_path" not in properties


async def test_each_agent_still_records_its_own_anchoring(
    pg, tmp_path, monkeypatch,
):
    """The per-agent facts did not vanish; they moved to where they belong.

    ``source_path`` is an operator filesystem path and ``verification`` is the
    result of checking the signature against *this* agent's resolved trust
    root. Both are recorded on the agent's own ``constitution_reanchor``
    property, which already carried them before #2893 — the node was
    duplicating them onto a row the fleet shares.
    """
    await _seed_runtime_agent(CONSTITUTION_V1, AGENT_DID)
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V2)
    import kestrel_sovereign.config as ks_config
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    artifact_path, root_path = _write_authority_files(tmp_path, CONSTITUTION_V2)
    agent_dir = tmp_path / "agent_data" / "First"
    _make_local_anchor(agent_dir, AGENT_DID)

    result = await reanchor_constitution(
        agent_name="First", agent_dir=agent_dir,
        canonical_path=constitution_path, force=True,
        amendment_artifact_path=artifact_path,
        sovereign_trust_root_path=root_path,
        runtime_backend="postgres", runtime_dsn=POSTGRES_URL,
    )
    assert result.error is None, result.error

    rows = await pg.fetchall(
        "SELECT properties FROM graph_nodes WHERE node_id = ?", (AGENT_DID,)
    )
    properties = json.loads(rows[0][0]) if isinstance(rows[0][0], str) else rows[0][0]
    audit = properties["constitution_reanchor"]
    assert audit["signed_artifact_path"] == str(artifact_path)
    assert audit["signed_artifact_verification"]
    assert audit["timestamp"]


async def test_a_per_agent_artifact_reanchors_the_second_agent(
    pg, tmp_path, monkeypatch,
):
    """The documented way through: the same constitution hash signed into a
    byte-distinct artifact. The authorization is identical; only the record is
    per-agent."""
    await _seed_runtime_agent(CONSTITUTION_V1, AGENT_DID)
    await _seed_runtime_agent(CONSTITUTION_V1, OTHER_DID)
    v2_hash = hashlib.sha256(CONSTITUTION_V2).hexdigest()
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V2)
    import kestrel_sovereign.config as ks_config
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))

    for did, name, reason in (
        (AGENT_DID, "First", "for First"),
        (OTHER_DID, "Second", "for Second"),
    ):
        root_path = tmp_path / "sovereign-root.did.json"
        root_path.write_text(
            json.dumps(
                did_document_from_legacy_public_key(
                    _ROOT_DID, _ROOT_KEYPAIR.public_key
                )
            ),
            encoding="utf-8",
        )
        artifact = build_legacy_signed_reanchor_artifact(
            signer_did=_ROOT_DID,
            constitution_sha256=v2_hash,
            private_key=_ROOT_KEYPAIR.private_key,
            reason=reason,
        )
        artifact_path = tmp_path / f"reanchor-{name}.signed.json"
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

        agent_dir = tmp_path / "agent_data" / name
        _make_local_anchor(agent_dir, did)
        result = await reanchor_constitution(
            agent_name=name,
            agent_dir=agent_dir,
            canonical_path=constitution_path,
            force=True,
            amendment_artifact_path=artifact_path,
            sovereign_trust_root_path=root_path,
            runtime_backend="postgres",
            runtime_dsn=POSTGRES_URL,
        )
        assert result.error is None, f"{name}: {result.error}"
        assert await _runtime_state(pg, did) == (v2_hash, [v2_hash])


async def test_overlay_anchor_lands_in_the_database_the_runtime_reads(
    pg, tmp_path, monkeypatch,
):
    """``anchor-overlay`` shares the target rule. Its property authorizes
    DANGEROUS Amendment IX grants, so writing it to a database the runtime
    never opens leaves every grant denied — while reporting success."""
    from kestrel_sovereign.setup.overlay_anchor import (
        OVERLAY_HASH_PROPERTY,
        anchor_overlay,
    )

    await _seed_runtime_agent(CONSTITUTION_V1)
    agent_dir = tmp_path / "agent_data" / "PgTargetAgent"
    anchor_bytes = _make_local_anchor(agent_dir, AGENT_DID)
    overlay = agent_dir / "CONSTITUTION.md"
    overlay.write_bytes(b"# Overlay\n\nshell_execution_host granted.\n")
    overlay_hash = hashlib.sha256(overlay.read_bytes()).hexdigest()

    result = await anchor_overlay(
        agent_name="PgTargetAgent",
        agent_dir=agent_dir,
        runtime_backend="postgres",
        runtime_dsn=POSTGRES_URL,
    )

    assert result.error is None, result.error
    assert result.new_hash == overlay_hash

    row = await pg.fetchone(
        "SELECT properties FROM graph_nodes WHERE node_id = $1", (AGENT_DID,)
    )
    properties = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    assert properties.get(OVERLAY_HASH_PROPERTY) == overlay_hash
    assert (agent_dir / "kestrel_prime.db").read_bytes() == anchor_bytes


async def test_reanchor_reads_drift_from_postgres_not_from_the_anchor(
    pg, tmp_path, monkeypatch,
):
    """Unforced. The runtime database is current, so the honest answer is
    "nothing to do" — reached by reading PostgreSQL. A run that consulted the
    anchor could not answer this question at all: the anchor holds no
    ``constitution_hash`` here."""
    v2_hash = await _seed_runtime_agent(CONSTITUTION_V2)
    agent_dir = tmp_path / "agent_data" / "PgTargetAgent"
    _make_local_anchor(agent_dir, AGENT_DID)

    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V2)
    import kestrel_sovereign.config as ks_config
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))

    result = await reanchor_constitution(
        agent_name="PgTargetAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=False,
        runtime_backend="postgres",
        runtime_dsn=POSTGRES_URL,
    )

    assert result.error is None, result.error
    assert result.unchanged is True
    assert result.old_hash == v2_hash
    assert result.target_backend == "postgres"


async def test_only_governed_by_edges_count_as_governance(pg, tmp_path, monkeypatch):
    """Real agents have several outgoing edges that are not governance:
    ``spawned_by`` on every spawned child, plus ``retired_via``,
    ``migrated_via``, ``has_avatar``. Without the ``label`` filter on the
    edge read, a reanchor treats them all as governing constitutions and
    reports the spawn parent's DID as a stale target to remove — permanently,
    because ``_write_reanchor`` keeps its own filter and so can never clear
    what the reader invented.
    """
    v1_hash = await _seed_runtime_agent(CONSTITUTION_V1, AGENT_DID)
    async with AsyncStorage(backend="postgres", dsn=POSTGRES_URL) as storage:
        storage.graph.bind_agent(AGENT_DID)
        await storage.graph.add_trusted_cross_agent_edge(
            AGENT_DID, OTHER_DID, "spawned_by"
        )

    agent_dir = tmp_path / "agent_data" / "PgTargetAgent"
    _make_local_anchor(agent_dir, AGENT_DID)
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V2)
    import kestrel_sovereign.config as ks_config
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))

    result = await reanchor_constitution(
        agent_name="PgTargetAgent",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=False,
        runtime_backend="postgres",
        runtime_dsn=POSTGRES_URL,
    )

    assert result.error is None, result.error
    # Only the previous constitution is stale. The spawn parent is not a
    # governing document and must never be offered up for deletion.
    assert result.stale_edge_targets == (v1_hash,)
    assert OTHER_DID not in result.stale_edge_targets
