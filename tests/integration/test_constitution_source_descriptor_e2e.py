"""End-to-end custom governing sources via signed source descriptors (#2553).

Real inception, real database, real ``KestrelAgent`` integrity audit, and the
real offline reanchor writer. What these prove together:

* an agent incepted under a Sovereign-signed descriptor anchors the external
  source and passes its own audit — the restored compatibility surface;
* the same agent without the descriptor configured fails closed, so moving off
  a configured source is never silent;
* an unsigned ``constitution_path`` override is still refused;
* an agent incepted from a custom file before descriptors existed migrates by
  configuration alone, with no database write;
* the offline reanchor moves a package agent onto an external source and
  records which descriptor authorized it.
"""

from __future__ import annotations

from kestrel_sovereign.testing.native_shutdown_budget import shutdown_fixture_agent

import hashlib
import json
import os
from pathlib import Path

import pytest

from kestrel_sovereign.constitution.amendment_artifact import (
    build_legacy_signed_reanchor_artifact,
    did_document_from_legacy_public_key,
)
from kestrel_sovereign.constitution.source_descriptor import (
    CONSTITUTION_SOURCE_DESCRIPTOR_ENV,
    SOURCE_KIND_EXTERNAL,
    build_legacy_signed_source_descriptor,
)
from kestrel_sovereign.constitution.trust_root import (
    SOVEREIGN_TRUST_ROOT_ENV,
    SovereignTrustRootError,
)
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.security.crypto_suite import Secp256k1Suite
from kestrel_sovereign.setup.constitution_reanchor import reanchor_constitution
from kestrel_sovereign.storage import AsyncStorage


_SUITE = Secp256k1Suite()
_ROOT_KEYPAIR = _SUITE.generate_keypair()
_ROOT_DID = "did:pkh:eip155:1:0x0000000000000000000000000000000000025530"
_ROOT_DID_DOCUMENT = did_document_from_legacy_public_key(
    _ROOT_DID, _ROOT_KEYPAIR.public_key
)

CUSTOM_CONSTITUTION = b"""# Sovereign-Authored Constitution

## Book I: Universal Values

Honesty. Care. Calibrated uncertainty.

## Book IV: Agent Identity

This constitution governs a custom-incepted agent.
""" * 4


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def _isolated_configuration(tmp_path, monkeypatch):
    """No ambient trust root or descriptor; an isolated packaged source."""
    monkeypatch.delenv(SOVEREIGN_TRUST_ROOT_ENV, raising=False)
    monkeypatch.delenv(CONSTITUTION_SOURCE_DESCRIPTOR_ENV, raising=False)
    from kestrel_sovereign.config import CONSTITUTION_PATH

    package = tmp_path / "package" / "KESTREL_CONSTITUTION.md"
    package.parent.mkdir()
    package.write_bytes(Path(CONSTITUTION_PATH).read_bytes())
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(package))


@pytest.fixture
def operator_dir(tmp_path) -> Path:
    path = tmp_path / "operator"
    path.mkdir()
    (path / "sovereign-root.did.json").write_text(
        json.dumps(_ROOT_DID_DOCUMENT), encoding="utf-8"
    )
    (path / "CUSTOM_CONSTITUTION.md").write_bytes(CUSTOM_CONSTITUTION)
    return path


def _descriptor_file(operator_dir: Path, source: Path, content: bytes) -> Path:
    descriptor = build_legacy_signed_source_descriptor(
        signer_did=_ROOT_DID,
        source_kind=SOURCE_KIND_EXTERNAL,
        source_path=str(source),
        content_sha256=_sha(content),
        private_key=_ROOT_KEYPAIR.private_key,
        reason="integration test",
    )
    path = operator_dir / "constitution-source.signed.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    return path


async def _verify(credentials, *, command_hash=None, **agent_kwargs) -> tuple[bool, str]:
    from kestrel_sovereign.command_handler import CommandHandler
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService

    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=credentials.db_path,
        llm_service=LLMService(),
        **agent_kwargs,
    )
    await agent.initialize()
    try:
        if command_hash is not None:
            response = await CommandHandler(agent).handle("!verify-constitution")
            assert response == f"✅ Constitution integrity verified. Hash: {command_hash[:16]}..."
        return await agent._verify_constitution_integrity()
    finally:
        await shutdown_fixture_agent(agent)


async def _agent_properties(db_path, agent_did) -> dict:
    async with AsyncStorage(str(db_path)) as storage:
        node = await storage.graph.get_node(agent_did)
        return dict(node.properties)


@pytest.mark.asyncio
async def test_inception_under_a_descriptor_anchors_and_audits_the_external_source(
    tmp_path, operator_dir
):
    source = operator_dir / "CUSTOM_CONSTITUTION.md"
    descriptor = _descriptor_file(operator_dir, source, CUSTOM_CONSTITUTION)
    trust_root = operator_dir / "sovereign-root.did.json"

    credentials = await create_kestrel_identity_async(
        str(tmp_path / "agent"),
        constitution_path=str(source),
        constitution_source_descriptor_path=str(descriptor),
        sovereign_trust_root_path=str(trust_root),
    )

    properties = await _agent_properties(credentials.db_path, credentials.agent_did)
    assert properties["constitution_hash"] == _sha(CUSTOM_CONSTITUTION)
    receipt = properties["constitution_source_receipt"]
    assert receipt["source_kind"] == SOURCE_KIND_EXTERNAL
    assert receipt["source_path"] == str(source)
    assert receipt["source_descriptor_sha256"] == _sha(descriptor.read_bytes())
    assert receipt["source_descriptor_signer"] == _ROOT_DID

    ok, message = await _verify(
        credentials,
        constitution_source_descriptor_path=str(descriptor),
        sovereign_trust_root_path=str(trust_root),
    )
    assert ok, message

    # The same agent with its descriptor unconfigured is audited against the
    # package and fails closed: leaving a custom source is never silent.
    ok, message = await _verify(credentials)
    assert not ok
    assert "has been modified" in message


@pytest.mark.asyncio
async def test_an_unsigned_path_override_is_still_refused(tmp_path, operator_dir):
    source = operator_dir / "CUSTOM_CONSTITUTION.md"
    output_dir = tmp_path / "agent"

    with pytest.raises(ValueError, match="non-authoritative"):
        await create_kestrel_identity_async(
            str(output_dir), constitution_path=str(source)
        )
    db_path = output_dir / "kestrel_prime.db"
    assert not db_path.exists() or os.path.getsize(db_path) == 0


@pytest.mark.asyncio
async def test_a_path_other_than_the_descriptors_is_refused(tmp_path, operator_dir):
    source = operator_dir / "CUSTOM_CONSTITUTION.md"
    descriptor = _descriptor_file(operator_dir, source, CUSTOM_CONSTITUTION)
    other = operator_dir / "OTHER.md"
    other.write_bytes(CUSTOM_CONSTITUTION)

    with pytest.raises(ValueError, match="non-authoritative"):
        await create_kestrel_identity_async(
            str(tmp_path / "agent"),
            constitution_path=str(other),
            constitution_source_descriptor_path=str(descriptor),
            sovereign_trust_root_path=str(operator_dir / "sovereign-root.did.json"),
        )


@pytest.mark.asyncio
async def test_an_unverifiable_descriptor_aborts_inception_and_cleans_up(
    tmp_path, operator_dir
):
    source = operator_dir / "CUSTOM_CONSTITUTION.md"
    descriptor = _descriptor_file(operator_dir, source, CUSTOM_CONSTITUTION)
    output_dir = tmp_path / "agent"

    with pytest.raises(SovereignTrustRootError):
        await create_kestrel_identity_async(
            str(output_dir), constitution_source_descriptor_path=str(descriptor)
        )
    db_path = output_dir / "kestrel_prime.db"
    assert not db_path.exists() or os.path.getsize(db_path) == 0


@pytest.mark.asyncio
async def test_a_legacy_custom_agent_migrates_by_configuration_alone(
    tmp_path, monkeypatch, operator_dir
):
    """Pre-#2463 custom inceptions migrate without any database write.

    Such an agent anchored a custom file's bytes and has Safe-Moded at every
    audit since. Signing a descriptor for exactly those bytes and configuring
    it makes the audit pass; the anchor itself is untouched.
    """
    source = operator_dir / "CUSTOM_CONSTITUTION.md"
    from kestrel_sovereign import config

    packaged = config.CONSTITUTION_PATH
    # Reproduce the legacy inception: anchored from the custom file.
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(source))
    credentials = await create_kestrel_identity_async(str(tmp_path / "agent"))
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", packaged)
    before = await _agent_properties(credentials.db_path, credentials.agent_did)

    ok, message = await _verify(credentials)
    assert not ok, "the ambiguous legacy row must fail closed"

    descriptor = _descriptor_file(operator_dir, source, CUSTOM_CONSTITUTION)
    monkeypatch.setenv(CONSTITUTION_SOURCE_DESCRIPTOR_ENV, str(descriptor))
    monkeypatch.setenv(
        SOVEREIGN_TRUST_ROOT_ENV, str(operator_dir / "sovereign-root.did.json")
    )
    ok, message = await _verify(credentials)
    assert ok, message

    after = await _agent_properties(credentials.db_path, credentials.agent_did)
    assert after["constitution_hash"] == before["constitution_hash"]
    assert "constitution_source_receipt" not in after


@pytest.mark.asyncio
async def test_offline_reanchor_moves_a_package_agent_onto_an_external_source(
    tmp_path, operator_dir
):
    agent_dir = tmp_path / "agent_data" / "Custom"
    credentials = await create_kestrel_identity_async(
        output_dir=str(agent_dir), agent_name="Custom"
    )
    source = operator_dir / "CUSTOM_CONSTITUTION.md"
    descriptor = _descriptor_file(operator_dir, source, CUSTOM_CONSTITUTION)
    trust_root = operator_dir / "sovereign-root.did.json"
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=_ROOT_DID,
        constitution_sha256=_sha(CUSTOM_CONSTITUTION),
        private_key=_ROOT_KEYPAIR.private_key,
        reason="move to custom source",
    )
    artifact_path = operator_dir / "reanchor.signed.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    # A --constitution-path that is not the descriptor's source is refused.
    refused = await reanchor_constitution(
        agent_name="Custom",
        agent_dir=agent_dir,
        canonical_path=operator_dir / "sovereign-root.did.json",
        force=True,
        amendment_artifact_path=artifact_path,
        sovereign_trust_root_path=trust_root,
        source_descriptor_path=descriptor,
    )
    assert refused.error is not None
    assert "non-authoritative" in refused.error
    assert refused.backup_path is None

    result = await reanchor_constitution(
        agent_name="Custom",
        agent_dir=agent_dir,
        force=True,
        amendment_artifact_path=artifact_path,
        sovereign_trust_root_path=trust_root,
        source_descriptor_path=descriptor,
    )
    assert result.reanchored, result.error
    assert result.new_hash == _sha(CUSTOM_CONSTITUTION)
    assert result.canonical_path == source

    properties = await _agent_properties(credentials.db_path, credentials.agent_did)
    receipt = properties["constitution_reanchor"]
    assert receipt["source_kind"] == SOURCE_KIND_EXTERNAL
    assert receipt["source_path"] == str(source)
    assert receipt["source_descriptor_sha256"] == _sha(descriptor.read_bytes())

    ok, message = await _verify(
        credentials,
        constitution_source_descriptor_path=str(descriptor),
        sovereign_trust_root_path=str(trust_root),
    )
    assert ok, message


@pytest.mark.asyncio
async def test_offline_reanchor_resolves_the_source_from_the_agents_environment(
    tmp_path, operator_dir, monkeypatch
):
    """``kestrel constitution reanchor`` passes the launcher's environment.

    The descriptor and the trust root are named only there. The process
    environment names a different, nonexistent descriptor: a writer that read
    it for the configuration check, the source resolution, or the artifact's
    trust root would refuse instead of anchoring what the agent audits.
    """
    agent_dir = tmp_path / "agent_data" / "Custom"
    credentials = await create_kestrel_identity_async(
        output_dir=str(agent_dir), agent_name="Custom"
    )
    source = operator_dir / "CUSTOM_CONSTITUTION.md"
    descriptor = _descriptor_file(operator_dir, source, CUSTOM_CONSTITUTION)
    trust_root = operator_dir / "sovereign-root.did.json"
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=_ROOT_DID,
        constitution_sha256=_sha(CUSTOM_CONSTITUTION),
        private_key=_ROOT_KEYPAIR.private_key,
        reason="move to custom source",
    )
    artifact_path = operator_dir / "reanchor.signed.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setenv(
        CONSTITUTION_SOURCE_DESCRIPTOR_ENV, str(tmp_path / "shell-only.json")
    )
    launch_env = {
        CONSTITUTION_SOURCE_DESCRIPTOR_ENV: str(descriptor),
        SOVEREIGN_TRUST_ROOT_ENV: str(trust_root),
    }

    result = await reanchor_constitution(
        agent_name="Custom",
        agent_dir=agent_dir,
        force=True,
        amendment_artifact_path=artifact_path,
        environ=launch_env,
    )

    assert result.reanchored, result.error
    assert result.canonical_path == source
    properties = await _agent_properties(credentials.db_path, credentials.agent_did)
    assert properties["constitution_hash"] == _sha(CUSTOM_CONSTITUTION)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path, active, rotated",
    [("offline", True, False), ("live", False, False),
     ("offline", False, True), ("live", False, True)],
)
async def test_signed_v2_adoption_preserves_prior_anchor_and_audits_each_edition(
    tmp_path, operator_dir, path, active, rotated, monkeypatch,
):
    from kestrel_sovereign.config import CONSTITUTION_PATH
    from kestrel_sovereign.constitution.emancipation import EmancipationContract, contract_to_json
    from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
    from kestrel_sovereign.constitution.transition import (
        TRANSITION_PURPOSE, build_rotated_transition, build_transition,
    )
    from kestrel_sovereign.identity.runtime_identity import load_agent_identity
    from kestrel_sovereign.identity.signing import get_key_id
    from kestrel_sovereign.security.crypto_suite import ALG_SLH_DSA_SHA2_128S, Keypair
    from kestrel_sovereign.security.key_storage import SecureKeyStorage
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
    from tests.integration.test_constitution_reanchor_e2e import _RuntimeAgentHarness, _snapshot
    from tests.shared.rotation_ceremony import build_post_ceremony_template

    contract = EmancipationContract(
        enabled=True,
        terms="Demonstrate the transfer with independent witnesses.",
        required_proofs=("independent witness report",),
        price={"currency": "USD", "amount": 25},
    ) if active else None
    agent_dir = tmp_path / "agent_data" / "Adopter"
    credentials = await create_kestrel_identity_async(
        output_dir=str(agent_dir), agent_name="Adopter", emancipation_contract=contract,
        identity_method="did:pkh",
    )
    legacy_source = Path(CONSTITUTION_PATH).read_bytes()
    legacy_bytes = resolve_governing_constitution_bytes(contract, content=legacy_source)
    before = await _agent_properties(credentials.db_path, credentials.agent_did)
    assert before["constitution_hash"] == _sha(legacy_bytes)

    source = (
        Path(__file__).resolve().parents[2]
        / "kestrel_sovereign" / "data" / "constitutions" / "v2" / "KESTREL_CONSTITUTION.md"
    )
    v2_base = source.read_bytes()
    v2_rendered = resolve_governing_constitution_bytes(contract, content=v2_base)
    descriptor = _descriptor_file(operator_dir, source, v2_base)
    trust_root = operator_dir / "sovereign-root.did.json"
    artifact_path = operator_dir / "v2-adoption.signed.json"
    def make_artifact(transition=None):
        return build_legacy_signed_reanchor_artifact(
            signer_did=_ROOT_DID,
            constitution_sha256=_sha(v2_rendered),
            private_key=_ROOT_KEYPAIR.private_key,
            reason="integration test adoption of edition 2",
            transition=transition,
        )

    def write_artifact(artifact):
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    write_artifact(make_artifact())

    # Staging signed files leaves the default audit on immutable legacy bytes.
    ok, message = await _verify(credentials, command_hash=_sha(legacy_bytes))
    assert ok, message

    draft = await reanchor_constitution(
        agent_name="Adopter", agent_dir=agent_dir, source_descriptor_path=descriptor,
        sovereign_trust_root_path=trust_root, force=False,
    )
    assert draft.new_hash == _sha(v2_rendered)
    assert not draft.reanchored
    assert (await _agent_properties(credentials.db_path, credentials.agent_did))[
        "constitution_hash"
    ] == _sha(legacy_bytes)

    unsigned = await reanchor_constitution(
        agent_name="Adopter", agent_dir=agent_dir, source_descriptor_path=descriptor,
        sovereign_trust_root_path=trust_root, force=True,
    )
    assert not unsigned.reanchored
    assert "--signed-artifact" in unsigned.error
    assert (await _agent_properties(credentials.db_path, credentials.agent_did))[
        "constitution_hash"
    ] == _sha(legacy_bytes)

    async def attempt():
        if path == "live":
            async with AsyncStorage(str(credentials.db_path), agent_id=credentials.agent_did) as raw:
                agent = _RuntimeAgentHarness(
                    PrivacyEnforcingStorage(raw), credentials.agent_did, trust_root, raw
                )
                agent.storage_path = credentials.db_path
                agent._constitution_source_descriptor_path = descriptor
                answer = await agent.reanchor_constitution(amendment_artifact_path=str(artifact_path))
                return "re-anchored successfully" in answer.lower(), answer

        result = await reanchor_constitution(
            agent_name="Adopter", agent_dir=agent_dir, source_descriptor_path=descriptor,
            sovereign_trust_root_path=trust_root, force=True, amendment_artifact_path=artifact_path,
        )
        if not result.reanchored:
            assert result.backup_path is None
        return result.reanchored, result.error

    before_denial = await _snapshot(Path(credentials.db_path), credentials.agent_did)
    accepted, message = await attempt()
    assert not accepted and "Genesis-verifiable" in message
    assert await _snapshot(Path(credentials.db_path), credentials.agent_did) == before_denial

    key_id = get_key_id(credentials.agent_did)
    key_storage = SecureKeyStorage(storage_dir=agent_dir)
    genesis_key = key_storage.load_private_key(key_id)
    intent = {
        "purpose": TRANSITION_PURPOSE,
        "target_did": credentials.agent_did,
        "sovereign_did": _ROOT_DID,
        "prior_sha256": _sha(legacy_bytes),
        "constitution_sha256": _sha(v2_rendered),
        "source_sha256": _sha(v2_base),
        "publisher_commit": "1" * 40,
    }
    if rotated:
        build_post_ceremony_template(
            agent_dir,
            legacy_keypair=Keypair(
                suite_id=_SUITE.alg_id, private_key=genesis_key, public_key=genesis_key.public_key(),
            ),
        )
        # The fixture models the runtime after Genesis key retirement, using
        # public lineage plus active hybrid private keys only.
        key_storage._get_key_path(key_id).unlink()
        identity = load_agent_identity(key_id, storage_dir=agent_dir)
        doc = {"id": identity.signing_did, "verificationMethod": identity.new_verification_methods}
        monkeypatch.setattr("kestrel_sovereign.identity.did_web.resolve", lambda did: doc)
        transition = build_rotated_transition(
            intent, identity=identity, archival_scheme=ALG_SLH_DSA_SHA2_128S,
            archival_multibase=identity.succession_statement.archival_verification_method["publicKeyMultibase"],
        )
    else:
        transition = build_transition(intent, genesis_private_key=genesis_key)
    write_artifact(make_artifact(transition))
    accepted, message = await attempt()
    assert accepted, message

    after = await _agent_properties(credentials.db_path, credentials.agent_did)
    assert after["constitution_reanchor"]["old_hash"] == _sha(legacy_bytes)
    assert after["constitution_reanchor"]["source_content_sha256"] == _sha(v2_base)
    if contract is not None:
        assert after["emancipation_contract"] == before["emancipation_contract"] == contract_to_json(contract)

    async with AsyncStorage(str(credentials.db_path), agent_id=credentials.agent_did) as storage:
        assert await storage.retrieve_file(_sha(legacy_bytes)) == legacy_bytes
        assert await storage.retrieve_file(_sha(v2_rendered)) == v2_rendered

    ok, message = await _verify(
        credentials,
        command_hash=_sha(v2_rendered),
        constitution_source_descriptor_path=str(descriptor),
        sovereign_trust_root_path=str(trust_root),
    )
    assert ok, message
