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
    """The guarantee is that the neighbour is never touched. That is unchanged.

    This test used to assert a *refusal*, and the refusal was a means to that
    end, not the end itself. With PostgreSQL holding only the neighbour, this
    agent's record is pending — boot will replicate its anchor — so anchoring
    the overlay locally is anchoring it where boot will look, exactly as the
    constitution reanchor does in the same state. What must never happen, and
    still does not, is the unbound read that grants the *neighbour* a
    DANGEROUS Amendment IX overlay.

    The refusal remains for the case it was written for: an agent that IS in
    the runtime database. There, the overlay hash is read from PostgreSQL, so
    writing it to the local file would report success while every grant stayed
    denied — the #2890 defect. That is covered below.
    """
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

    assert result.error is None, result.error
    assert result.target_backend == "sqlite", result.target_label

    # The point of the test: the neighbour is untouched.
    row = await pg.fetchone(
        "SELECT properties FROM graph_nodes WHERE node_id = $1", (OTHER_DID,)
    )
    neighbour = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    assert OVERLAY_HASH_PROPERTY not in neighbour


async def test_overlay_anchor_still_refuses_for_a_replicated_agent(pg, tmp_path):
    """The #2890 case, unchanged: an agent PostgreSQL already holds.

    Its overlay hash is read from the runtime database, so writing it to the
    local file would report success while every Amendment IX grant stayed
    denied. Only a *pending* record redirects to the anchor.
    """
    from kestrel_sovereign.setup.overlay_anchor import anchor_overlay

    await _seed_runtime_agent(CONSTITUTION_V1, AGENT_DID)
    agent_dir = tmp_path / "agent_data" / "PgTargetAgent"
    _make_local_anchor(agent_dir, AGENT_DID)
    (agent_dir / "CONSTITUTION.md").write_bytes(b"# Overlay\n\ngranted.\n")

    result = await anchor_overlay(
        agent_name="PgTargetAgent",
        agent_dir=agent_dir,
        runtime_backend="postgres",
        runtime_dsn=POSTGRES_URL,
    )

    assert result.error is None, result.error
    assert result.target_backend == "postgres", result.target_label


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


async def test_reusing_one_signed_artifact_across_agents_refuses_legibly(
    pg, tmp_path, monkeypatch,
):
    """A signed artifact is content-addressed, so one file is one graph node
    for the whole database. The second agent cannot claim a node the first
    owns — ``add_node`` allows a foreign-owned content node only for the
    constitution document, and the artifact node carries ``source_path``,
    which the privacy boundary does not treat as content-free.

    That limitation is real (#2893). What must not happen is the operator
    seeing an opaque mid-write rollback: the first agent's reanchor committed,
    the second one's did not, and nothing said why or what to do.
    """
    await _seed_runtime_agent(CONSTITUTION_V1, AGENT_DID)
    other_v1 = await _seed_runtime_agent(CONSTITUTION_V1, OTHER_DID)
    v2_hash = hashlib.sha256(CONSTITUTION_V2).hexdigest()
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V2)
    import kestrel_sovereign.config as ks_config
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    artifact_path, root_path = _write_authority_files(tmp_path, CONSTITUTION_V2)

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

    assert results["First"].error is None, results["First"].error
    assert await _runtime_state(pg, AGENT_DID) == (v2_hash, [v2_hash])

    refusal = results["Second"].error
    assert refusal is not None
    assert "already anchored" in refusal
    assert AGENT_DID in refusal
    assert "Sign a separate artifact" in refusal
    assert "#2893" in refusal
    # Refused before the write, not rolled back after a partial one.
    assert "Cannot overwrite a graph node" not in refusal
    assert results["Second"].backup_path is None
    assert await _runtime_state(pg, OTHER_DID) == (other_v1, [other_v1])


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


async def test_a_forced_reanchor_repairs_an_unwitnessed_governed_by_edge(
    pg, tmp_path, monkeypatch,
):
    """Detecting the drift is only half a remedy.

    A physical ``governed_by`` edge at the right hash with no
    ``graph_edge_owners`` witness is invisible to the bound store, so integrity
    proof 2 fails and the agent safe-modes. ``add_edge`` also refuses to claim
    an edge nobody owns, so the forced repair rolled its whole transaction back
    — doctor prescribing a command that could never clear the finding it
    raised. The writer now drops the witness-less row first and lets
    ``add_edge`` recreate it with its ledger entry.
    """
    await _seed_runtime_agent(CONSTITUTION_V1, AGENT_DID)
    v1_hash = hashlib.sha256(CONSTITUTION_V1).hexdigest()
    # Strip the witness, leaving the physical edge: the damaged state.
    await pg.execute(
        "DELETE FROM graph_edge_owners WHERE source_id = $1 AND label = 'governed_by'",
        (AGENT_DID,),
    )

    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V1)
    import kestrel_sovereign.config as ks_config
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    artifact_path, root_path = _write_authority_files(tmp_path, CONSTITUTION_V1)
    agent_dir = tmp_path / "agent_data" / "First"
    _make_local_anchor(agent_dir, AGENT_DID)

    result = await reanchor_constitution(
        agent_name="First",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=True,
        amendment_artifact_path=artifact_path,
        sovereign_trust_root_path=root_path,
        runtime_backend="postgres",
        runtime_dsn=POSTGRES_URL,
    )

    assert result.error is None, result.error

    owners = await pg.fetchall(
        "SELECT agent_id FROM graph_edge_owners "
        "WHERE source_id = $1 AND target_id = $2 AND label = 'governed_by'",
        (AGENT_DID, v1_hash),
    )
    assert [row[0] for row in owners] == [AGENT_DID], (
        "the repaired edge still has no ownership witness, so proof 2 still fails"
    )


async def test_a_placeholder_runtime_record_sends_the_repair_to_the_anchor(
    pg, tmp_path, monkeypatch,
):
    """A boot-fabricated stand-in is pending, not present.

    Doctor already treats it that way — ``birth_record`` counts it as an
    identity shortfall and boot replaces it from the anchor before auditing —
    and prescribes this command for drift in those pending bytes. Reading the
    placeholder as a real record made the reanchor answer "no constitution_hash"
    and leave the stale anchor that boot then replicates and safe-modes on: the
    same finding and remedy disagreeing about which bytes they mean.
    """
    v1_hash = hashlib.sha256(CONSTITUTION_V1).hexdigest()
    # The exact shape `_ensure_agent_node_present` writes.
    await pg.execute(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
        "VALUES ($1, 'agent', $2, $3)",
        (AGENT_DID, f"Agent {AGENT_DID}", json.dumps({"initialBalance": "100.0"})),
    )
    await pg.execute(
        "INSERT INTO graph_node_owners (node_id, agent_id) VALUES ($1, $2)",
        (AGENT_DID, AGENT_DID),
    )

    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION_V2)
    import kestrel_sovereign.config as ks_config
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    agent_dir = tmp_path / "agent_data" / "First"
    _make_local_anchor(agent_dir, AGENT_DID)
    # Give the anchor a real anchored hash: it is the record boot will copy,
    # and the drift under test lives in it.
    import sqlite3
    with sqlite3.connect(str(agent_dir / "kestrel_prime.db")) as conn:
        conn.execute(
            "UPDATE graph_nodes SET properties = ? WHERE node_id = ?",
            (json.dumps({"name": "First", "constitution_hash": v1_hash}), AGENT_DID),
        )
        conn.commit()

    result = await reanchor_constitution(
        agent_name="First",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=False,
        runtime_backend="postgres",
        runtime_dsn=POSTGRES_URL,
    )

    assert result.target_backend == "sqlite", result.target_label
    assert result.old_hash == v1_hash, result.error


async def test_a_pending_overlay_is_anchored_where_boot_will_find_it(
    pg, tmp_path, monkeypatch,
):
    """The third tool has to follow the same rule as the other two.

    With PostgreSQL not yet holding this agent, doctor reports on the anchor
    and prescribes ``kestrel constitution anchor-overlay``. Resolving that
    command against PostgreSQL — where there is no agent node — failed without
    touching the anchor, so first boot copied an unanchored overlay and
    safe-moded. Pending is the one state where the local file is the runtime's
    future contents.
    """
    from kestrel_sovereign.setup.overlay_anchor import (
        OVERLAY_HASH_PROPERTY,
        anchor_overlay,
    )

    agent_dir = tmp_path / "agent_data" / "First"
    _make_local_anchor(agent_dir, AGENT_DID)
    (agent_dir / "CONSTITUTION.md").write_bytes(b"# Overlay\n\npending.\n")

    result = await anchor_overlay(
        agent_name="First",
        agent_dir=agent_dir,
        runtime_backend="postgres",
        runtime_dsn=POSTGRES_URL,
    )

    assert result.error is None, result.error
    async with AsyncStorage(str(agent_dir / "kestrel_prime.db"),
                            backend="sqlite", agent_id=AGENT_DID) as storage:
        agent = await storage.graph.get_node(AGENT_DID)
    assert agent.properties.get(OVERLAY_HASH_PROPERTY) == result.new_hash


async def test_a_foreign_owner_row_is_not_pending_replication(
    pg, tmp_path, monkeypatch,
):
    """Ledger damage can wear the shape of a state boot repairs.

    An *absent* ``graph_nodes`` row whose ownership ledger still carries a
    foreign witness looks exactly like a clean pending replication — and is
    not: ``add_node`` refuses a row owned by another agent, so boot cannot
    land the copy. Calling it pending would retarget the repair to the local
    anchor and leave PostgreSQL unusable while reporting success.
    """
    from kestrel_sovereign.setup.constitution_reanchor import (
        runtime_record_is_pending,
        resolve_reanchor_target,
    )

    await pg.execute(
        "INSERT INTO graph_node_owners (node_id, agent_id) VALUES ($1, $2)",
        (AGENT_DID, OTHER_DID),
    )
    agent_dir = tmp_path / "agent_data" / "PgTargetAgent"
    _make_local_anchor(agent_dir, AGENT_DID)

    target = await resolve_reanchor_target(
        agent_dir, backend="postgres", dsn=POSTGRES_URL
    )

    assert await runtime_record_is_pending(target) is False


async def test_a_genuinely_absent_record_is_still_pending(pg, tmp_path):
    """The veto is about *conflicting* ownership, not about caution."""
    from kestrel_sovereign.setup.constitution_reanchor import (
        runtime_record_is_pending,
        resolve_reanchor_target,
    )

    agent_dir = tmp_path / "agent_data" / "PgTargetAgent"
    _make_local_anchor(agent_dir, AGENT_DID)

    target = await resolve_reanchor_target(
        agent_dir, backend="postgres", dsn=POSTGRES_URL
    )

    assert await runtime_record_is_pending(target) is True
