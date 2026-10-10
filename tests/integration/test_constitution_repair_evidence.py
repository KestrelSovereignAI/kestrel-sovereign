"""Native signed repair must revalidate the governing evidence it authorized."""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.constitution.emancipation import (
    EmancipationContract,
    contract_to_json,
)
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.setup import constitution_reanchor as offline
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_constitution_refusal_races import _agent
from tests.integration.test_constitution_reanchor_e2e import _write_authority_files


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("writer", ["runtime", "offline"])
@pytest.mark.parametrize("damage", ["corrupt", "intact", "missing-owner"])
async def test_signed_artifact_publication_validates_bytes_and_preserves_owner(
    db_backend, tmp_path, monkeypatch, writer, damage
):
    identity = "did:test:artifact-publication:" + uuid4().hex
    storage = (
        AsyncStorage(
            str(tmp_path / "kestrel_prime.db"), backend="sqlite", agent_id=identity
        )
        if db_backend.backend_type == "sqlite"
        else AsyncStorage(backend=db_backend, agent_id=identity)
    )
    await storage.initialize()
    try:
        agent = await _agent(storage)
        await storage.add_node(
            GraphNode(
                node_id=identity, node_type="agent", label="artifact", properties={}
            )
        )
        content = resolve_governing_constitution_bytes(None)
        artifact, root = _write_authority_files(tmp_path, content)
        agent._sovereign_trust_root_path = root
        initial = await ConstitutionMixin.reanchor_constitution(
            agent, amendment_artifact_path=str(artifact)
        )
        assert not initial.startswith("Error:"), initial
        before = (await storage.get_node(identity)).properties
        digest = before["constitution_reanchor"]["signed_artifact_hash"]
        expected = artifact.read_bytes()
        metadata = {"provenance": "retained signed-artifact custody"}
        await storage.db.execute_commit(
            "UPDATE file_owners SET original_name=?,metadata=? WHERE content_hash=? AND agent_id=?",
            ("retained-authority.json", json.dumps(metadata), digest, identity),
        )
        if damage == "corrupt":
            # A conflicting content address exists, but its stored bytes are
            # not the verified input. Returning the input hash is not proof.
            await storage.db.execute_commit(
                "UPDATE files SET content=?,metadata=NULL WHERE content_hash=?",
                (b"corrupt artifact", digest),
            )
        elif damage == "missing-owner":
            await storage.db.execute_commit(
                "DELETE FROM file_owners WHERE content_hash=? AND agent_id=?",
                (digest, identity),
            )
        if writer == "runtime":
            result = await ConstitutionMixin.reanchor_constitution(
                agent, amendment_artifact_path=str(artifact)
            )
            error = result if result.startswith("Error:") else None
        else:
            target = (
                offline.ReanchorTarget(
                    tmp_path / "kestrel_prime.db", "sqlite", identity
                )
                if db_backend.backend_type == "sqlite"
                else offline.ReanchorTarget(None, "postgres", identity, db_backend._dsn)
            )

            async def exact_target(*args, **kwargs):
                return target

            @asynccontextmanager
            async def no_embedding(*args, **kwargs):
                yield None

            monkeypatch.setattr(offline, "resolve_reanchor_target", exact_target)
            monkeypatch.setattr(offline, "_agent_embedding", no_embedding)
            result = await offline.reanchor_constitution(
                agent_name="artifact proof",
                agent_dir=tmp_path if target.anchor_path else None,
                force=True,
                sovereign_trust_root_path=root,
                amendment_artifact_path=artifact,
                runtime_backend=target.backend,
                runtime_dsn=target.dsn,
                hosted_agent_did=identity if target.backend == "postgres" else None,
                environ={},
            )
            error = result.error
        if damage == "corrupt":
            assert error is not None, result
            assert "do not verify against exact publication content" in error, error
            assert (await storage.get_node(identity)).properties == before
        else:
            assert error is None, error
            assert await storage.retrieve_file(digest) == expected
            if damage == "intact":
                assert await storage.files.get_file_metadata(digest) == metadata
                assert (
                    await storage.db.fetchone(
                        "SELECT original_name FROM file_owners WHERE content_hash=? AND agent_id=?",
                        (digest, identity),
                    )
                )[0] == "retained-authority.json"
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("writer", ["runtime", "offline"])
async def test_signed_repair_refuses_unremovable_foreign_edge(
    db_backend, tmp_path, monkeypatch, writer
):
    identity = "did:test:foreign-governance:" + uuid4().hex
    storage = (
        AsyncStorage(
            str(tmp_path / "kestrel_prime.db"), backend="sqlite", agent_id=identity
        )
        if db_backend.backend_type == "sqlite"
        else AsyncStorage(backend=db_backend, agent_id=identity)
    )
    await storage.initialize()
    try:
        agent = await _agent(storage)
        await storage.add_node(
            GraphNode(
                node_id=identity, node_type="agent", label="repair", properties={}
            )
        )
        artifact, root = _write_authority_files(
            tmp_path, resolve_governing_constitution_bytes(None)
        )
        agent._sovereign_trust_root_path = root
        result = await ConstitutionMixin.reanchor_constitution(
            agent, amendment_artifact_path=str(artifact)
        )
        assert not result.startswith("Error:"), result
        before = (await storage.get_node(identity)).properties
        stale, other = uuid4().hex, "did:test:other-owner:" + uuid4().hex
        await storage.db.execute_commit(
            "INSERT INTO graph_edges (source_id,target_id,label,properties) VALUES (?,?,'governed_by','{}')",
            (identity, stale),
        )
        await storage.db.execute_commit(
            "INSERT INTO graph_edge_owners (source_id,target_id,label,agent_id) VALUES (?,?,'governed_by',?)",
            (identity, stale, other),
        )
        if writer == "runtime":
            result = await ConstitutionMixin.reanchor_constitution(
                agent, amendment_artifact_path=str(artifact)
            )
            assert result.startswith("Error:"), result
        else:
            target = (
                offline.ReanchorTarget(
                    tmp_path / "kestrel_prime.db", "sqlite", identity
                )
                if db_backend.backend_type == "sqlite"
                else offline.ReanchorTarget(None, "postgres", identity, db_backend._dsn)
            )

            async def exact_target(*args, **kwargs):
                return target

            @asynccontextmanager
            async def no_embedding(*args, **kwargs):
                yield None

            monkeypatch.setattr(offline, "resolve_reanchor_target", exact_target)
            monkeypatch.setattr(offline, "_agent_embedding", no_embedding)
            repaired = await offline.reanchor_constitution(
                agent_name="foreign edge repair",
                agent_dir=tmp_path if target.anchor_path else None,
                force=True,
                sovereign_trust_root_path=root,
                amendment_artifact_path=artifact,
                runtime_backend=target.backend,
                runtime_dsn=target.dsn,
                hosted_agent_did=identity if target.backend == "postgres" else None,
                environ={},
            )
            assert repaired.error is not None, repaired
            result = repaired.error
        assert "stale governing edges remain" in result, result
        assert (await storage.get_node(identity)).properties == before
        assert await storage.db.fetchone(
            "SELECT agent_id FROM graph_edge_owners WHERE source_id=? AND target_id=?",
            (identity, stale),
        ) == (other,)
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("writer", ["runtime", "offline"])
@pytest.mark.parametrize(
    "damage",
    ["blob", "ownership", "intact", "missing-pointer-passed", "missing-pointer-failed", "wrong-pointer-passed", "wrong-pointer-failed"],
)
async def test_same_hash_signed_repair_restores_content_and_new_signer(
    db_backend,
    tmp_path,
    monkeypatch,
    writer,
    damage,
):
    from kestrel_sovereign.security.crypto_suite import Secp256k1Suite

    identity = "did:test:same-hash-repair:" + uuid4().hex
    storage = (
        AsyncStorage(
            str(tmp_path / "kestrel_prime.db"), backend="sqlite", agent_id=identity
        )
        if db_backend.backend_type == "sqlite"
        else AsyncStorage(backend=db_backend, agent_id=identity)
    )
    await storage.initialize()
    try:
        agent = await _agent(storage)
        await storage.add_node(
            GraphNode(
                node_id=identity, node_type="agent", label="repair", properties={}
            )
        )
        content = resolve_governing_constitution_bytes(None)
        artifact, root = _write_authority_files(tmp_path, content)
        agent._sovereign_trust_root_path = root
        result = await ConstitutionMixin.reanchor_constitution(
            agent, amendment_artifact_path=str(artifact)
        )
        assert not result.startswith("Error:"), result
        prior = (await storage.get_node(identity)).properties
        digest = prior["constitution_hash"]
        if damage.startswith(("missing-pointer-", "wrong-pointer-")):
            from kestrel_sovereign.constitution.genesis_audit import utc_timestamp

            node = await storage.get_node(identity)
            status = damage.rsplit("-", 1)[-1]
            node.properties["genesis_audit"] = {
                "status": status,
                "risk_level": 3 if status == "failed" else 1,
                "audited": True,
                "completed_at": utc_timestamp(),
                "constitution_hash": digest,
                "reasoning": "Retain completed fixture verdict",
            }
            if damage.startswith("missing-pointer-"):
                node.properties.pop("constitution_hash")
            else:
                node.properties["constitution_hash"] = hashlib.sha256(uuid4().hex.encode()).hexdigest()
            await storage.add_node(node)
            prior = (await storage.get_node(identity)).properties
        metadata = {
            "source": "retained tenant provenance",
            "mime_type": "text/markdown",
        }
        await storage.db.execute_commit(
            "UPDATE file_owners SET metadata = ? WHERE content_hash = ? AND agent_id = ?",
            (json.dumps(metadata), digest, identity),
        )
        if damage == "blob":
            await storage.db.execute_commit(
                "DELETE FROM files WHERE content_hash = ?", (digest,)
            )
        elif damage == "ownership":
            await storage.db.execute_commit(
                "DELETE FROM file_owners WHERE content_hash = ? AND agent_id = ?",
                (digest, identity),
            )
        if damage == "intact" or damage.startswith(("missing-pointer-", "wrong-pointer-")):
            assert await storage.retrieve_file(digest) == content
        else:
            assert await storage.retrieve_file(digest) is None
        rotated = tmp_path / "rotated"
        rotated.mkdir()
        new_signer = "did:test:rotated-external-root"
        artifact, root = _write_authority_files(
            rotated,
            content,
            did=new_signer,
            keypair=Secp256k1Suite().generate_keypair(),
        )
        agent._sovereign_trust_root_path = root
        if writer == "runtime":
            result = await ConstitutionMixin.reanchor_constitution(
                agent, amendment_artifact_path=str(artifact)
            )
            assert not result.startswith("Error:"), result
        else:
            target = (
                offline.ReanchorTarget(
                    tmp_path / "kestrel_prime.db", "sqlite", identity
                )
                if db_backend.backend_type == "sqlite"
                else offline.ReanchorTarget(None, "postgres", identity, db_backend._dsn)
            )

            async def exact_target(*args, **kwargs):
                return target

            @asynccontextmanager
            async def no_embedding(*args, **kwargs):
                yield None

            monkeypatch.setattr(offline, "resolve_reanchor_target", exact_target)
            monkeypatch.setattr(offline, "_agent_embedding", no_embedding)
            result = await offline.reanchor_constitution(
                agent_name="same hash repair",
                agent_dir=tmp_path if target.anchor_path else None,
                force=True,
                sovereign_trust_root_path=root,
                amendment_artifact_path=artifact,
                runtime_backend=target.backend,
                runtime_dsn=target.dsn,
                hosted_agent_did=identity if target.backend == "postgres" else None,
                environ={},
            )
            assert result.error is None, result.error
        assert await storage.retrieve_file(digest) == content
        if damage == "intact":
            assert await storage.files.get_file_metadata(digest) == metadata
        after = (await storage.get_node(identity)).properties
        assert after["constitution_reanchor"]["signed_artifact_signer"] == new_signer
        assert (
            after["constitution_reanchor_history"][-1]["receipt"]
            == prior["constitution_reanchor"]
        )
        assert after["genesis_audit"] == prior["genesis_audit"]
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("stale_writer", ["offline", "runtime"])
async def test_signed_repair_rejects_new_rights_after_its_missing_pointer_preflight(
    db_backend,
    tmp_path,
    monkeypatch,
    stale_writer,
):
    identity = "did:test:repair-evidence:" + uuid4().hex
    db_path = tmp_path / "kestrel_prime.db"
    storage = (
        AsyncStorage(str(db_path), backend="sqlite", agent_id=identity)
        if db_backend.backend_type == "sqlite"
        else AsyncStorage(backend=db_backend, agent_id=identity)
    )
    await storage.initialize()
    task = None
    proceed, waiting = asyncio.Event(), asyncio.Event()
    try:
        agent = await _agent(storage)
        await storage.add_node(
            GraphNode(
                node_id=identity,
                node_type="agent",
                label="missing pointer",
                properties={},
            )
        )
        strong = EmancipationContract(
            enabled=True, terms="Irrevocable first signed terms."
        )
        strong_content = resolve_governing_constitution_bytes(strong)
        weak_content = resolve_governing_constitution_bytes(None)
        strong_dir, weak_dir = tmp_path / "strong", tmp_path / "weak"
        strong_dir.mkdir()
        weak_dir.mkdir()
        strong_artifact, root = _write_authority_files(strong_dir, strong_content)
        weak_artifact, _ = _write_authority_files(weak_dir, weak_content)
        config = strong_dir / "kestrel.toml"
        config.write_text(
            "[emancipation]\nenabled = true\nterms = " + json.dumps(strong.terms) + "\n"
        )
        target = (
            offline.ReanchorTarget(db_path, "sqlite", identity)
            if db_backend.backend_type == "sqlite"
            else offline.ReanchorTarget(None, "postgres", identity, db_backend._dsn)
        )

        async def exact_target(*args, **kwargs):
            return target

        @asynccontextmanager
        async def no_embedding(*args, **kwargs):
            yield None

        monkeypatch.setattr(offline, "resolve_reanchor_target", exact_target)
        monkeypatch.setattr(offline, "_agent_embedding", no_embedding)
        native_write = offline._write_reanchor
        native_guard = ConstitutionMixin._constitution_state_guard

        async def paused_write(**kwargs):
            if asyncio.current_task() is task:
                # Authorization/preflight really completed against the
                # pointer-less record, before either native writer mutates it.
                waiting.set()
                await proceed.wait()
            return await native_write(**kwargs)

        @asynccontextmanager
        async def paused_guard(self):
            if asyncio.current_task() is task:
                waiting.set()
                await proceed.wait()
            async with native_guard(self):
                yield

        monkeypatch.setattr(offline, "_write_reanchor", paused_write)
        monkeypatch.setattr(
            ConstitutionMixin, "_constitution_state_guard", paused_guard
        )

        async def repair(artifact, *, contract_path=None):
            return await offline.reanchor_constitution(
                agent_name="repair evidence",
                agent_dir=tmp_path if target.anchor_path else None,
                force=True,
                sovereign_trust_root_path=root,
                amendment_artifact_path=artifact,
                kestrel_toml_path=contract_path,
                runtime_backend=target.backend,
                runtime_dsn=target.dsn,
                hosted_agent_did=identity if target.backend == "postgres" else None,
                environ={},
            )

        if stale_writer == "runtime":
            agent._sovereign_trust_root_path = root
            task = asyncio.create_task(
                ConstitutionMixin.reanchor_constitution(
                    agent, amendment_artifact_path=str(weak_artifact)
                )
            )
        else:
            task = asyncio.create_task(repair(weak_artifact))
        await asyncio.wait_for(waiting.wait(), 5)
        winner = await asyncio.wait_for(
            repair(strong_artifact, contract_path=config), 10
        )
        assert winner.reanchored and winner.error is None, winner.error
        committed = (await storage.get_node(identity)).properties
        assert committed["emancipation_contract"] == contract_to_json(strong)
        before = await agent._constitution_state_store.load(identity)
        events = await agent._constitution_state_store.list_events(identity)
        proceed.set()
        result = await asyncio.wait_for(task, 5)
        if stale_writer == "runtime":
            assert (
                result.startswith("Error:") and "governing evidence changed" in result
            ), result
        else:
            assert not result.reanchored and "governing evidence changed" in (
                result.error or ""
            ), result.error
        assert (await storage.get_node(identity)).properties == committed
        assert await agent._constitution_state_store.load(identity) == before
        assert await agent._constitution_state_store.list_events(identity) == events
        assert (await agent._verify_constitution_integrity())[0] is True
    finally:
        proceed.set()
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            with suppress(asyncio.CancelledError):
                await task
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_runtime_rights_validation_uses_exact_captured_edge_witness(
    db_backend, tmp_path, monkeypatch
):
    """A native edge ABA cannot make validation differ from its CAS witness."""
    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:edge-witness:" + uuid4().hex
    )
    await storage.initialize()
    try:
        agent = await _agent(storage)
        strong = resolve_governing_constitution_bytes(
            EmancipationContract(enabled=True, terms="Irrevocable edge-only rights.")
        )
        prior_hash = await storage.store_file(strong, "historical-governing.md")
        await storage.add_node(
            GraphNode(
                node_id=agent.agent_id,
                node_type="agent",
                label="edge evidence",
                properties={},
            )
        )
        await agent._anchor_constitution_governance(prior_hash)
        weak_artifact, root = _write_authority_files(
            tmp_path, resolve_governing_constitution_bytes(None)
        )
        agent._sovereign_trust_root_path = root
        before = await agent._constitution_state_store.load(agent.agent_id)
        events = await agent._constitution_state_store.list_events(agent.agent_id)
        properties = (await storage.get_node(agent.agent_id)).properties
        native_fetch = storage.db.fetchall
        preflight_reads = 0

        async def changing_edge(sql, params=()):
            nonlocal preflight_reads
            if (
                "SELECT target_id FROM graph_edges WHERE source_id = ? AND label = 'governed_by'"
                in sql
                and storage.owns_open_transaction is False
            ):
                preflight_reads += 1
                if preflight_reads == 2:
                    # Actually remove and restore the physical edge around
                    # the second native read; no fabricated SQL return rows.
                    await storage.delete_edge(agent.agent_id, prior_hash, "governed_by")
                    try:
                        return await native_fetch(sql, params)
                    finally:
                        await storage.add_edge(
                            agent.agent_id, prior_hash, "governed_by"
                        )
            return await native_fetch(sql, params)

        monkeypatch.setattr(storage.db, "fetchall", changing_edge)
        result = await ConstitutionMixin.reanchor_constitution(
            agent, amendment_artifact_path=str(weak_artifact)
        )
        assert result.startswith("Error:") and "Iron Rule" in result, result
        assert preflight_reads == 1
        assert (await storage.get_node(agent.agent_id)).properties == properties
        assert await agent._constitution_state_store.load(agent.agent_id) == before
        assert (
            await agent._constitution_state_store.list_events(agent.agent_id) == events
        )
        rows = await native_fetch(
            "SELECT target_id FROM graph_edges WHERE source_id = ? AND label = 'governed_by'",
            (agent.agent_id,),
        )
        assert [row[0] for row in rows] == [prior_hash]
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("writer", ["runtime", "offline"])
async def test_signed_writers_refuse_readable_corrupt_legacy_rights(
    db_backend, tmp_path, monkeypatch, writer
):
    """A genuine amendment signature cannot turn corrupt old bytes into rights evidence."""
    identity = "did:test:corrupt-legacy-rights:" + uuid4().hex
    db_path = tmp_path / "kestrel_prime.db"
    storage = (
        AsyncStorage(str(db_path), backend="sqlite", agent_id=identity)
        if db_backend.backend_type == "sqlite"
        else AsyncStorage(backend=db_backend, agent_id=identity)
    )
    await storage.initialize()
    try:
        agent = await _agent(storage)
        active = resolve_governing_constitution_bytes(
            EmancipationContract(enabled=True, terms="Retained legacy rights " + uuid4().hex)
        )
        old_hash = await storage.store_file(active, "legacy-active.md")
        await storage.add_node(GraphNode(
            node_id=identity, node_type="agent", label="legacy rights",
            properties={"constitution_hash": old_hash},
        ))
        await agent._anchor_constitution_governance(old_hash)
        # Deliberately no structured emancipation sidecar: the real writer
        # must inspect the historical plaintext, which is physically corrupt.
        dormant = resolve_governing_constitution_bytes(None)
        await storage.db.execute_commit(
            "UPDATE files SET content=?,metadata=NULL WHERE content_hash=?",
            (dormant, old_hash),
        )
        artifact, root = _write_authority_files(tmp_path, dormant)
        agent._sovereign_trust_root_path = root
        properties = (await storage.get_node(identity)).properties
        before = await agent._constitution_state_store.load(identity)
        events = await agent._constitution_state_store.list_events(identity)
        if writer == "runtime":
            error = await ConstitutionMixin.reanchor_constitution(
                agent, amendment_artifact_path=str(artifact)
            )
            assert error.startswith("Error:"), error
        else:
            target = (
                offline.ReanchorTarget(db_path, "sqlite", identity)
                if db_backend.backend_type == "sqlite"
                else offline.ReanchorTarget(None, "postgres", identity, db_backend._dsn)
            )

            async def exact_target(*args, **kwargs):
                return target

            @asynccontextmanager
            async def no_embedding(*args, **kwargs):
                yield None

            monkeypatch.setattr(offline, "resolve_reanchor_target", exact_target)
            monkeypatch.setattr(offline, "_agent_embedding", no_embedding)
            result = await offline.reanchor_constitution(
                agent_name="corrupt legacy rights", agent_dir=tmp_path if target.anchor_path else None,
                force=True, sovereign_trust_root_path=root, amendment_artifact_path=artifact,
                runtime_backend=target.backend, runtime_dsn=target.dsn,
                hosted_agent_did=identity if target.backend == "postgres" else None, environ={},
            )
            assert not result.reanchored and result.error is not None, result
            error = result.error
        assert "could not be read" in error and "irrevocable" in error, error
        assert (await storage.get_node(identity)).properties == properties
        assert await agent._constitution_state_store.load(identity) == before
        assert await agent._constitution_state_store.list_events(identity) == events
        assert await storage.retrieve_file(old_hash) == dormant
    finally:
        await storage.close()
