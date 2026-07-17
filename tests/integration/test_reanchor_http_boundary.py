"""Authenticated HTTP regression for the live reanchor trust boundary (#2499)."""

from __future__ import annotations

import asyncio
import hashlib
import json

from fastapi.testclient import TestClient

from kestrel_sovereign.constitution.amendment_artifact import (
    build_legacy_signed_reanchor_artifact,
    did_document_from_legacy_public_key,
)
from kestrel_sovereign.bootstrap import BootstrapState
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.security.crypto_suite import Secp256k1Suite
from kestrel_sovereign.server import app
from kestrel_sovereign.storage import AsyncStorage


V1 = b"# HTTP trust-boundary constitution v1\n"
V2 = b"# HTTP trust-boundary constitution v2\n"


async def _mutate_graph_root(db_path, agent_did, attacker_doc):
    async with AsyncStorage(str(db_path)) as storage:
        agent = await storage.graph.get_node(agent_did)
        agent.properties.update(
            {
                "constitution_hash": "db-writer-replaced-hash",
                "sovereign_root_did_document": attacker_doc,
                "trusted_sovereign_did_document": attacker_doc,
                "sovereign_root_did": attacker_doc["id"],
                "sovereign_root_public_key_hex": attacker_doc["publicKey"][0][
                    "publicKeyHex"
                ],
            }
        )
        await storage.graph.add_node(agent)


async def _governance_snapshot(db_path, agent_did):
    async with AsyncStorage(str(db_path)) as storage:
        agent = await storage.graph.get_node(agent_did)
        edges = await storage.graph.get_edges(agent_did, direction="out")
        documents = await storage.graph.get_nodes_by_type("document")
        artifacts = await storage.graph.get_nodes_by_type(
            "constitution_amendment_artifact"
        )
        return {
            "properties": agent.properties,
            "governed_by": sorted(
                edge.target_id for edge in edges if edge.label == "governed_by"
            ),
            "documents": sorted(node.node_id for node in documents),
            "artifacts": sorted(node.node_id for node in artifacts),
        }


def test_authenticated_http_rejects_db_injected_root_without_mutation(
    tmp_path, monkeypatch,
):
    """Drive the real command handler through `/api/agent/invoke`."""
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(V1)
    import kestrel_sovereign.config as ks_config

    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    agent_dir = tmp_path / "kite"
    creds = asyncio.run(
        create_kestrel_identity_async(
            output_dir=str(agent_dir),
            constitution_path=str(constitution_path),
            agent_name="Kite",
        )
    )
    db_path = agent_dir / "kestrel_prime.db"
    constitution_path.write_bytes(V2)

    suite = Secp256k1Suite()
    legitimate_keypair = suite.generate_keypair()
    legitimate_did = "did:example:legitimate-sovereign"
    legitimate_doc = did_document_from_legacy_public_key(
        legitimate_did,
        legitimate_keypair.public_key,
    )
    trust_root_path = tmp_path / "legitimate-root.did.json"
    trust_root_path.write_text(json.dumps(legitimate_doc), encoding="utf-8")

    attacker_keypair = suite.generate_keypair()
    attacker_did = "did:example:db-attacker"
    attacker_doc = did_document_from_legacy_public_key(
        attacker_did,
        attacker_keypair.public_key,
    )
    asyncio.run(_mutate_graph_root(db_path, creds.agent_did, attacker_doc))
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=attacker_did,
        constitution_sha256=hashlib.sha256(V2).hexdigest(),
        private_key=attacker_keypair.private_key,
        reason="forged graph-root HTTP probe",
    )
    artifact_path = tmp_path / "attacker-reanchor.signed.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    legitimate_artifact = build_legacy_signed_reanchor_artifact(
        signer_did=legitimate_did,
        constitution_sha256=hashlib.sha256(V2).hexdigest(),
        private_key=legitimate_keypair.private_key,
        reason="legitimate HTTP reanchor",
    )
    legitimate_artifact_path = tmp_path / "legitimate-reanchor.signed.json"
    legitimate_artifact_path.write_text(
        json.dumps(legitimate_artifact),
        encoding="utf-8",
    )
    before = asyncio.run(_governance_snapshot(db_path, creds.agent_did))

    monkeypatch.setenv("KESTREL_DB_PATH", str(agent_dir))
    monkeypatch.setenv("KESTREL_API_KEY", "kite-test-api-key")
    monkeypatch.setenv("KESTREL_SYNC_ENABLED", "false")
    monkeypatch.setenv(
        "KESTREL_SOVEREIGN_TRUST_ROOT_PATH",
        str(trust_root_path),
    )

    with TestClient(app) as client:
        client.portal.call(
            app.state.agent.bootstrap_service.set_bootstrap_state,
            BootstrapState.COMPLETE,
        )
        app.state.agent._safe_mode = True
        response = client.post(
            "/api/agent/invoke",
            json={
                "input": (
                    f"!reanchor-constitution {artifact_path} "
                    f"{hashlib.sha256(V2).hexdigest()[:8]}"
                )
            },
            headers={"X-API-Key": "kite-test-api-key"},
        )
        assert response.status_code == 200, response.text
        body = response.json()["response"]
        assert "Signed amendment verification failed" in body
        assert "not trusted Sovereign DID" in body
        assert app.state.agent._safe_mode is True

        rejected = client.portal.call(
            _governance_snapshot,
            db_path,
            creds.agent_did,
        )
        assert rejected == before

        valid_response = client.post(
            "/api/agent/invoke",
            json={
                "input": (
                    f"!reanchor-constitution {legitimate_artifact_path} "
                    f"{hashlib.sha256(V2).hexdigest()[:8]}"
                )
            },
            headers={"X-API-Key": "kite-test-api-key"},
        )
        assert valid_response.status_code == 200, valid_response.text
        valid_body = valid_response.json()["response"]
        assert "re-anchored successfully" in valid_body.lower()
        assert app.state.agent._safe_mode is True

    after = asyncio.run(_governance_snapshot(db_path, creds.agent_did))
    v2_hash = hashlib.sha256(V2).hexdigest()
    assert after["properties"]["constitution_hash"] == v2_hash
    assert after["properties"]["constitution_reanchor"][
        "signed_artifact_signer"
    ] == legitimate_did
    assert v2_hash in after["governed_by"]
    assert after["artifacts"]
    assert list(agent_dir.glob("*.backup-*")) == []

    # A full application lifespan restart must load the newly authorized
    # anchor from durable storage rather than falling back to the injected DB
    # root or the old hash.
    with TestClient(app) as restarted_client:
        persisted_node = restarted_client.portal.call(
            app.state.agent.storage.get_node,
            creds.agent_did,
        )
        assert persisted_node.properties["constitution_hash"] == v2_hash
