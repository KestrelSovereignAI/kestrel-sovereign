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

import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path
import threading

import pytest

from kestrel_sovereign.constitution.publisher import verify_packaged_publisher
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
@pytest.mark.parametrize("autocrlf", [False, True], ids=["lf", "crlf"])
async def test_existing_packaged_agent_keeps_its_original_checkout_bytes(tmp_path, autocrlf):
    from kestrel_sovereign.config import CONSTITUTION_PATH
    from kestrel_sovereign.constitution.v1_prior import recover_published_v1

    root = Path(__file__).resolve().parents[2]
    path = "HEAD:kestrel_sovereign/data/KESTREL_CONSTITUTION.md"
    original = subprocess.check_output(["git", "cat-file", "blob", path], cwd=root)
    if autocrlf:
        original = original.replace(b"\n", b"\r\n")
    digest = _sha(original)
    package = Path(CONSTITUTION_PATH)
    package.write_bytes(original)
    identity = await create_kestrel_identity_async(output_dir=str(tmp_path / "agent"))
    ok, message = await _verify(identity, command_hash=digest)
    assert ok, message
    before = await _agent_properties(identity.db_path, identity.agent_did)
    assert before["constitution_hash"] == digest

    checkout = subprocess.check_output([
        "git", "-c", f"core.autocrlf={str(autocrlf).lower()}", "cat-file", "--filters", path,
    ], cwd=root)
    package.write_bytes(checkout)
    ok, message = await _verify(identity, command_hash=digest)
    assert ok, message
    assert checkout == original
    assert recover_published_v1(digest).encode() == original
    after = await _agent_properties(identity.db_path, identity.agent_did)
    assert after["constitution_hash"] == before["constitution_hash"]
    assert after["genesis_audit"] == before["genesis_audit"]


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
    content = _legacy_role_revision(source)
    descriptor = _descriptor_file(operator_dir, source, content)
    trust_root = operator_dir / "sovereign-root.did.json"
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=_ROOT_DID,
        constitution_sha256=_sha(content),
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
    assert result.new_hash == _sha(content)
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
    content = _legacy_role_revision(source)
    descriptor = _descriptor_file(operator_dir, source, content)
    trust_root = operator_dir / "sovereign-root.did.json"
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=_ROOT_DID,
        constitution_sha256=_sha(content),
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
    assert properties["constitution_hash"] == _sha(content)


def _legacy_role_revision(source: Path) -> bytes:
    from kestrel_sovereign.config import CONSTITUTION_PATH

    original = Path(CONSTITUTION_PATH).read_bytes()
    position = original.index(b"\n## The Amendment Process")
    content = original[:position] + b"\nThe agent's assigned role is research support.\n" + original[position:]
    source.write_bytes(content)
    return content


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ["offline", "live"])
@pytest.mark.parametrize(
    "revision", [
        "current-book-ii", "book-iv", "historical-book-ii", "historical-book-ii-missing",
        "joint-intermediate-publication", "joint-current-publication", "signed-frame-preamble", "signed-frame-procedure",
    ],
)
async def test_native_ranked_v1_amendment_uses_published_authority(
    tmp_path, operator_dir, writer, revision,
):
    from kestrel_sovereign.config import CONSTITUTION_PATH
    from kestrel_sovereign.constitution.emancipation import (
        EmancipationContract, apply_emancipation, contract_from_json,
    )
    from kestrel_sovereign.constitution.genesis_audit import utc_timestamp
    from kestrel_sovereign.constitution.legacy_authority import LEGACY_FIXED_AMENDMENT_PURPOSE
    from kestrel_sovereign.constitution.transition import build_transition
    from kestrel_sovereign.constitution.v1_releases import CURRENT_V1_RELEASE, V1_RELEASES
    from kestrel_sovereign.identity.signing import get_key_id
    from kestrel_sovereign.security.key_storage import SecureKeyStorage
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
    from tests.integration.test_constitution_reanchor_e2e import _RuntimeAgentHarness, _snapshot

    prior_prefix = "763e6ac2c" if revision.startswith("historical-book-ii") else "84f4591ab"
    if revision.startswith("signed-frame-"):
        prior_prefix = "f0672b394"
    prior = next(release for release in V1_RELEASES if release.commit.startswith(prior_prefix))
    prior_bytes = prior.authenticated_text().encode()
    Path(CONSTITUTION_PATH).write_bytes(prior_bytes)
    if revision == "signed-frame-preamble":
        candidate = prior_bytes.replace(b"AI Agents", b"AI agents", 1)
    elif revision == "signed-frame-procedure":
        candidate = prior_bytes.replace(b"This Constitution can be amended according to", b"This Constitution may be amended according to", 1)
    elif revision.startswith("historical-book-ii"):
        later = next(release for release in V1_RELEASES if release.commit.startswith("0b24719bc"))
        candidate = later.authenticated_text().encode()
    elif revision == "joint-intermediate-publication":
        later = next(release for release in V1_RELEASES if release.commit.startswith("f0672b394"))
        candidate = later.authenticated_text().encode()
    elif revision == "book-iv":
        candidate = prior_bytes.replace(
            b"\n## Article V: The Amendment Process",
            b"\nThe agent may organize its research notes.\n\n## Article V: The Amendment Process",
        )
    else:
        candidate = CURRENT_V1_RELEASE.authenticated_text().encode()
    assert candidate != prior_bytes

    agreement = None
    if revision.startswith("joint-"):
        agreement = EmancipationContract(enabled=True, terms="Complete the transfer rehearsal.")
    at_birth = agreement if writer == "live" else None
    prior_hash = _sha(apply_emancipation(prior.authenticated_text(), at_birth).encode())
    config = tmp_path / "kestrel.toml"
    if agreement is not None:
        config.write_text(f'[emancipation]\nenabled = true\nterms = {json.dumps(agreement.terms)}\n')
    agent_dir = tmp_path / "agent_data" / "Ranked"
    credentials = await create_kestrel_identity_async(
        output_dir=str(agent_dir), agent_name="Ranked", identity_method="did:pkh", emancipation_contract=at_birth,
    )
    source = operator_dir / "ranked-constitution.md"
    source.write_bytes(candidate)
    descriptor = _descriptor_file(operator_dir, source, candidate)
    candidate = apply_emancipation(candidate.decode(), agreement).encode()
    trust_root = operator_dir / "sovereign-root.did.json"
    artifact = operator_dir / "v1-amendment.signed.json"

    def sign(declaration=None):
        artifact.write_text(json.dumps(build_legacy_signed_reanchor_artifact(
            signer_did=_ROOT_DID, constitution_sha256=_sha(candidate),
            private_key=_ROOT_KEYPAIR.private_key, legacy_fixed_amendment=declaration,
        )), encoding="utf-8")

    async def attempt():
        if writer == "offline":
            result = await reanchor_constitution(
                agent_name="Ranked", agent_dir=agent_dir, source_descriptor_path=descriptor,
                sovereign_trust_root_path=trust_root, force=True, amendment_artifact_path=artifact, kestrel_toml_path=config,
            )
            return result.reanchored, result.error
        async with AsyncStorage(str(credentials.db_path), agent_id=credentials.agent_did) as raw:
            agent = _RuntimeAgentHarness(
                PrivacyEnforcingStorage(raw), credentials.agent_did, trust_root, raw,
            )
            agent.storage_path = credentials.db_path
            agent._constitution_source_descriptor_path = descriptor
            result = await agent.reanchor_constitution(amendment_artifact_path=str(artifact))
        return "re-anchored successfully" in result.lower(), result

    if revision == "historical-book-ii-missing":
        async with AsyncStorage(str(credentials.db_path), agent_id=credentials.agent_did) as raw:
            await raw.db.execute("DELETE FROM files WHERE content_hash = ?", (prior.digest,))

    sign()
    if revision != "book-iv":
        before = await _snapshot(Path(credentials.db_path), credentials.agent_did)
        accepted, message = await attempt()
        assert not accepted and "Genesis-verifiable declaration" in message
        assert await _snapshot(Path(credentials.db_path), credentials.agent_did) == before
        genesis = SecureKeyStorage(storage_dir=agent_dir).load_private_key(get_key_id(credentials.agent_did))
        sign(build_transition({
            "purpose": LEGACY_FIXED_AMENDMENT_PURPOSE,
            "target_did": credentials.agent_did,
            "sovereign_did": _ROOT_DID,
            "prior_sha256": prior_hash,
            "constitution_sha256": _sha(candidate),
            "admission_at": utc_timestamp(),
        }, genesis_private_key=genesis))

    accepted, message = await attempt()
    assert accepted, message
    properties = await _agent_properties(credentials.db_path, credentials.agent_did)
    assert properties["constitution_hash"] == _sha(candidate)
    if agreement is not None:
        assert contract_from_json(properties["emancipation_contract"]) == agreement

    if revision == "signed-frame-procedure":
        previous_hash = _sha(candidate)
        candidate = candidate.replace(b"AI Agents", b"AI agents", 1)
        source.write_bytes(candidate)
        descriptor = _descriptor_file(operator_dir, source, candidate)
        sign(build_transition({
            "purpose": LEGACY_FIXED_AMENDMENT_PURPOSE,
            "target_did": credentials.agent_did,
            "sovereign_did": _ROOT_DID,
            "prior_sha256": previous_hash,
            "constitution_sha256": _sha(candidate),
            "admission_at": utc_timestamp(),
        }, genesis_private_key=genesis))
        accepted, message = await attempt()
        assert accepted, message

    if revision == "book-iv":
        # A second role edit needs the first signed revision to establish which
        # governing sections the new edit actually preserves.
        current = candidate
        candidate = current.replace(b"organize its research notes", b"organize its reading notes")
        source.write_bytes(candidate)
        descriptor = _descriptor_file(operator_dir, source, candidate)
        sign()
        async with AsyncStorage(str(credentials.db_path), agent_id=credentials.agent_did) as raw:
            await raw.db.execute("DELETE FROM files WHERE content_hash = ?", (_sha(current),))
        before = await _snapshot(Path(credentials.db_path), credentials.agent_did)
        accepted, message = await attempt()
        assert not accepted and "Restore the exact prior governing bytes" in message
        assert await _snapshot(Path(credentials.db_path), credentials.agent_did) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("at_birth", [True, False], ids=["grandfathered-terms", "activate-after-role"])
async def test_historical_role_and_personal_agreement_are_independent(tmp_path, operator_dir, at_birth):
    from kestrel_sovereign.config import CONSTITUTION_PATH
    from kestrel_sovereign.constitution.emancipation import EmancipationContract, apply_emancipation, contract_from_json
    from kestrel_sovereign.constitution.v1_releases import V1_RELEASES

    release = next(item for item in V1_RELEASES if item.commit.startswith("84f4591ab"))
    prior = release.authenticated_text()
    Path(CONSTITUTION_PATH).write_bytes(prior.encode())
    role = prior.replace("## Book IV: Agent Identity\n", "## Book IV: Agent Identity\n\nKeep the research journal.\n")
    contract = EmancipationContract(enabled=True, terms=(
        "Quote the headings in the record.\n\n```markdown\n### Amendment VIII: Emancipation\n## Book IV: Agent Identity\n```"
        if at_birth else "Complete the agreed transfer rehearsal."
    ))
    agent_dir = tmp_path / "Historian"
    signed = operator_dir / "role-change.json"
    config = tmp_path / "kestrel.toml"
    credentials = await create_kestrel_identity_async(
        output_dir=str(agent_dir), agent_name="Historian", identity_method="did:pkh",
        emancipation_contract=contract if at_birth else None,
    )
    source = operator_dir / "role.md"
    source.write_bytes(role.encode())
    descriptor = _descriptor_file(operator_dir, source, role.encode())

    async def revise(personal):
        digest = _sha(apply_emancipation(role, personal).encode())
        signed.write_text(json.dumps(build_legacy_signed_reanchor_artifact(
            signer_did=_ROOT_DID, private_key=_ROOT_KEYPAIR.private_key, constitution_sha256=digest,
        )), encoding="utf-8")
        result = await reanchor_constitution(
            agent_name="Historian", agent_dir=agent_dir, force=True, source_descriptor_path=descriptor,
            sovereign_trust_root_path=operator_dir / "sovereign-root.did.json", amendment_artifact_path=signed,
            kestrel_toml_path=config,
        )
        assert result.reanchored, result.error
        properties = await _agent_properties(credentials.db_path, credentials.agent_did)
        assert properties["constitution_hash"] == digest
        if personal:
            assert contract_from_json(properties["emancipation_contract"]) == personal

    await revise(contract if at_birth else None)
    if at_birth:
        from kestrel_sovereign.constitution.v1_prior import verify_supported_v1_prior

        personal_role = apply_emancipation(role, contract)
        evidence = verify_supported_v1_prior(
            _sha(personal_role.encode()), contract, prior_text=personal_role,
            prior_role_artifact_bytes=signed.read_bytes(), trusted_did_document=_ROOT_DID_DOCUMENT,
        )
        assert evidence["prior_role_artifact_sha256"] == _sha(signed.read_bytes())
    if not at_birth:
        config.write_text(f'[emancipation]\nenabled = true\nterms = {json.dumps(contract.terms)}\n')
        await revise(contract)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path, active, rotated, legacy_release",
    [("offline", True, False, None), ("live", False, False, None),
     ("offline", False, True, None), ("live", False, True, None),
     ("offline", True, False, "84f4591ab"), ("live", True, False, "f0672b394")],
)
async def test_signed_v2_adoption_preserves_prior_anchor_and_audits_each_edition(
    tmp_path, operator_dir, path, active, rotated, legacy_release, monkeypatch,
):
    from kestrel_sovereign.config import CONSTITUTION_PATH
    from kestrel_sovereign.constitution.emancipation import EmancipationContract, contract_to_json
    from kestrel_sovereign.constitution.genesis_audit import utc_timestamp
    from kestrel_sovereign.constitution.grants import (
        build_grant_baseline, build_offline_grant_observation, preview_offline_legacy_grants,
    )
    from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
    from kestrel_sovereign.constitution.transition import (
        TRANSITION_PURPOSE, build_rotated_transition, build_transition,
        publisher_source_url, verify_archived_transition_adoption,
    )
    from kestrel_sovereign.identity.runtime_identity import load_agent_identity
    from kestrel_sovereign.identity.signing import get_key_id
    from kestrel_sovereign.security.crypto_suite import ALG_SLH_DSA_SHA2_128S, Keypair
    from kestrel_sovereign.security.key_storage import SecureKeyStorage
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
    from tests.integration.test_constitution_reanchor_e2e import _RuntimeAgentHarness, _snapshot
    from tests.shared.rotation_ceremony import build_post_ceremony_template

    from kestrel_sovereign.constitution.v1_releases import CURRENT_V1_RELEASE, V1_RELEASES

    published_prior = next(
        release for release in V1_RELEASES
        if release.commit.startswith(legacy_release or CURRENT_V1_RELEASE.commit)
    )
    legacy_source = published_prior.authenticated_text().encode()
    Path(CONSTITUTION_PATH).write_bytes(legacy_source)
    contract = EmancipationContract(
        enabled=True,
        terms=(
            "Quote the heading in the record.\n\n```markdown\n### Amendment VIII: Emancipation\n```"
            if legacy_release == "84f4591ab" else "Demonstrate the transfer with independent witnesses."
        ),
        required_proofs=("independent witness report",),
        price={"currency": "USD", "amount": 25},
    ) if active else None
    monkeypatch.setenv("KESTREL_CAPABILITY_GRANTS", "filesystem_read")
    agent_dir = tmp_path / "agent_data" / "Adopter"
    credentials = await create_kestrel_identity_async(
        output_dir=str(agent_dir), agent_name="Adopter", emancipation_contract=contract,
        identity_method="did:pkh",
    )
    legacy_bytes = resolve_governing_constitution_bytes(contract, content=legacy_source)
    historical_release = next(
        release for release in V1_RELEASES if release.supports_activation and release != published_prior
    )
    historical_bytes = resolve_governing_constitution_bytes(
        contract, content=historical_release.authenticated_text().encode(),
    )
    historical_digest = _sha(historical_bytes)
    async with AsyncStorage(str(credentials.db_path), agent_id=credentials.agent_did) as storage:
        assert await storage.retrieve_file(historical_digest) is None
        node = await storage.get_node(credentials.agent_did)
        node.properties["constitution_reanchor"] = {"new_hash": historical_digest}
        await storage.add_node(node)
    before = await _agent_properties(credentials.db_path, credentials.agent_did)
    assert before["constitution_hash"] == _sha(legacy_bytes)

    source = (
        Path(__file__).resolve().parents[2]
        / "kestrel_sovereign" / "data" / "constitutions" / "v2" / "KESTREL_CONSTITUTION.md"
    )
    v2_base = source.read_bytes()
    publisher = verify_packaged_publisher(_sha(v2_base))
    v2_rendered = resolve_governing_constitution_bytes(contract, content=v2_base)
    descriptor = _descriptor_file(operator_dir, source, v2_base)
    trust_root = operator_dir / "sovereign-root.did.json"
    artifact_path = operator_dir / "v2-adoption.signed.json"
    observed = preview_offline_legacy_grants(agent_dir, overlay_hash=None)
    grant_baseline = build_grant_baseline(credentials.agent_did, observed)
    observation = build_offline_grant_observation(
        target_did=credentials.agent_did, prior_sha256=_sha(legacy_bytes),
        observed_at=utc_timestamp(), resolution=observed,
        workspace_policy={"enabled": False},
    )

    def make_artifact(transition=None, *, offline_observation=True):
        return build_legacy_signed_reanchor_artifact(
            signer_did=_ROOT_DID,
            constitution_sha256=_sha(v2_rendered),
            private_key=_ROOT_KEYPAIR.private_key,
            reason="integration test adoption of edition 2",
            transition=transition,
            grant_baseline=grant_baseline,
            publisher_release=publisher,
            offline_grant_observation=observation if offline_observation else None,
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
                accepted = (
                    "re-anchored successfully" in answer.lower()
                    or "Signed repair restored content/ownership" in answer
                )
                if accepted:
                    assert agent._constitution_host_grant_baseline.verified_governing_anchor == (
                        credentials.agent_did, _sha(v2_rendered),
                    )
                return accepted, answer

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
        "publisher_commit": publisher["publisher_commit"],
        "publisher_source_url": publisher_source_url(publisher["publisher_commit"]),
        "admission_at": utc_timestamp(),
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
        published_raw = json.dumps(doc)
        monkeypatch.setattr(
            "kestrel_sovereign.identity.did_web.resolve_document",
            lambda did: (published_raw.encode(), doc),
        )
        transition = build_rotated_transition(
            intent, identity=identity, archival_scheme=ALG_SLH_DSA_SHA2_128S,
            published_did_document=published_raw,
            archival_multibase=identity.succession_statement.archival_verification_method["publicKeyMultibase"],
        )
    else:
        transition = build_transition(intent, genesis_private_key=genesis_key)
    if path == "offline":
        write_artifact(make_artifact(transition, offline_observation=False))
        accepted, message = await attempt()
        assert not accepted and "pre-stop grant observation" in message
        assert await _snapshot(Path(credentials.db_path), credentials.agent_did) == before_denial

    write_artifact(make_artifact(transition))
    accepted, message = await attempt()
    assert accepted, message

    after = await _agent_properties(credentials.db_path, credentials.agent_did)
    assert after["constitution_reanchor"]["old_hash"] == _sha(legacy_bytes)
    assert after["constitution_reanchor"]["source_content_sha256"] == _sha(v2_base)
    assert after["constitution_reanchor_history"][-1]["receipt"]["new_hash"] == historical_digest
    if contract is not None:
        assert after["emancipation_contract"] == before["emancipation_contract"] == contract_to_json(contract)

    async with AsyncStorage(str(credentials.db_path), agent_id=credentials.agent_did) as storage:
        assert await storage.retrieve_file(_sha(legacy_bytes)) == legacy_bytes
        assert await storage.retrieve_file(_sha(v2_rendered)) == v2_rendered
        receipt = after["constitution_reanchor"]
        retained_artifact = await storage.retrieve_file(receipt["signed_artifact_hash"])
        assert retained_artifact == artifact_path.read_bytes()

        if rotated:
            from kestrel_sovereign.constitution.amendment_artifact import AmendmentArtifactError
            from kestrel_sovereign.constitution.grants import load_anchored_grant_baseline

            valid = await load_anchored_grant_baseline(
                storage.db, target_did=credentials.agent_did, properties=after,
                trusted_did_document=_ROOT_DID_DOCUMENT,
            )
            assert valid.grants == frozenset({"filesystem_read"})
            damaged_receipt = dict(receipt)
            del damaged_receipt["genesis_admission_evidence"]
            with pytest.raises(AmendmentArtifactError, match="recorded admission custody"):
                await load_anchored_grant_baseline(
                    storage.db, target_did=credentials.agent_did,
                    properties=after | {"constitution_reanchor": damaged_receipt},
                    trusted_did_document=_ROOT_DID_DOCUMENT,
                )

    def forbid_network(*args, **kwargs):
        pytest.fail("Replay of native proof consulted the live publication")

    with monkeypatch.context() as replay:
        replay.setattr("kestrel_sovereign.identity.did_web.resolve_document", forbid_network)
        verify_archived_transition_adoption(
            artifact_bytes=retained_artifact, receipt=receipt,
            trusted_did_document=_ROOT_DID_DOCUMENT, target_did=credentials.agent_did,
            prior_hash=_sha(legacy_bytes), prior_text=legacy_bytes.decode(),
            anchored_contract=contract, new_text=v2_rendered.decode(),
            new_hash=_sha(v2_rendered), source_hash=_sha(v2_base),
        )

    ok, message = await _verify(
        credentials,
        command_hash=_sha(v2_rendered),
        constitution_source_descriptor_path=str(descriptor),
        sovereign_trust_root_path=str(trust_root),
    )
    assert ok, message

    if active or rotated:
        governing_hash = _sha(v2_rendered)
        receipt_history = after.get("constitution_reanchor_history")
        for missing in (
            (governing_hash,), (receipt["signed_artifact_hash"],),
            (governing_hash, receipt["signed_artifact_hash"]),
        ):
            async with AsyncStorage(str(credentials.db_path), agent_id=credentials.agent_did) as storage:
                for digest in missing:
                    await storage.db.execute("DELETE FROM files WHERE content_hash = ?", (digest,))
            if receipt["signed_artifact_hash"] in missing:
                before_refusal = await _snapshot(Path(credentials.db_path), credentials.agent_did)
                write_artifact(make_artifact(offline_observation=False))
                accepted, message = await attempt()
                assert not accepted and "baseline artifact is unavailable" in message
                assert await _snapshot(Path(credentials.db_path), credentials.agent_did) == before_refusal

            artifact_path.write_bytes(retained_artifact)
            accepted, message = await attempt()
            assert accepted, message
            repaired = await _agent_properties(credentials.db_path, credentials.agent_did)
            assert repaired["constitution_reanchor"] == receipt
            assert repaired.get("constitution_reanchor_history") == receipt_history
            async with AsyncStorage(str(credentials.db_path), agent_id=credentials.agent_did) as storage:
                assert await storage.retrieve_file(governing_hash) == v2_rendered
                assert await storage.retrieve_file(receipt["signed_artifact_hash"]) == retained_artifact

    if rotated:
        # The original classical key is gone. A repair may use the retained
        # native adoption evidence without declaring the same adoption again.
        # A second process may already be verifying the old receipt while a
        # request awaits approval. Both readers must observe the committed repair.
        from kestrel_sovereign.constitution import grants

        async with AsyncStorage(str(credentials.db_path), agent_id=credentials.agent_did) as observer:
            agent = _RuntimeAgentHarness(PrivacyEnforcingStorage(observer), credentials.agent_did, trust_root, observer)
            node = await observer.get_node(credentials.agent_did)
            ok, message = await agent._refresh_constitution_host_grants(node, governing_text=v2_rendered.decode())
            assert ok, message
            admitted = agent._constitution_host_grant_baseline
            assert await grants.host_grant_record_is_current(agent, admitted)
            checked, resume = asyncio.Event(), asyncio.Event()
            verify = grants.load_anchored_grant_baseline

            async def pause_after_verification(db, **kwargs):
                result = await verify(db, **kwargs)
                if db is observer.db:
                    checked.set()
                    await resume.wait()
                return result

            with monkeypatch.context() as during_replay:
                during_replay.setattr(grants, "load_anchored_grant_baseline", pause_after_verification)
                replay = asyncio.create_task(agent._refresh_constitution_host_grants(node, governing_text=v2_rendered.decode()))
                try:
                    await asyncio.wait_for(checked.wait(), timeout=10)
                    async with AsyncStorage(str(credentials.db_path), agent_id=credentials.agent_did) as writer:
                        await writer.db.execute("DELETE FROM files WHERE content_hash = ?", (_sha(v2_rendered),))
                    write_artifact(make_artifact(offline_observation=False))
                    accepted, message = await attempt()
                    assert accepted, message
                    assert agent._constitution_host_grant_baseline is admitted
                    assert not await grants.host_grant_record_is_current(agent, admitted)
                finally:
                    resume.set()
                    valid, reason = await replay
                assert not valid and "Governing record changed" in reason
                assert agent._constitution_governing_replay_pending
            fresh = await observer.get_node(credentials.agent_did)
            valid, reason = await agent._refresh_constitution_host_grants(fresh, governing_text=v2_rendered.decode())
            assert valid, reason
            assert await grants.host_grant_record_is_current(agent, agent._constitution_host_grant_baseline)
        repaired = await _agent_properties(credentials.db_path, credentials.agent_did)
        assert repaired["constitution_reanchor"]["old_hash"] == _sha(v2_rendered)
        assert repaired["constitution_reanchor_history"][-1]["receipt"] == receipt
        async with AsyncStorage(str(credentials.db_path), agent_id=credentials.agent_did) as storage:
            assert await storage.retrieve_file(_sha(v2_rendered)) == v2_rendered
            retained = await load_anchored_grant_baseline(
                storage.db, target_did=credentials.agent_did, properties=repaired,
                trusted_did_document=_ROOT_DID_DOCUMENT,
            )
            assert retained.grants == frozenset({"filesystem_read"})


@pytest.mark.asyncio
async def test_new_v2_genesis_uses_signed_source_without_prior_adoption(tmp_path, operator_dir):
    source = (
        Path(__file__).resolve().parents[2]
        / "kestrel_sovereign" / "data" / "constitutions" / "v2" / "KESTREL_CONSTITUTION.md"
    )
    digest = _sha(source.read_bytes())
    publisher = verify_packaged_publisher(digest)
    descriptor = _descriptor_file(operator_dir, source, source.read_bytes())
    trust_root = operator_dir / "sovereign-root.did.json"
    credentials = await create_kestrel_identity_async(
        output_dir=str(tmp_path / "new-v2"),
        constitution_source_descriptor_path=str(descriptor),
        sovereign_trust_root_path=str(trust_root),
    )
    properties = await _agent_properties(credentials.db_path, credentials.agent_did)
    assert properties["constitution_hash"] == digest
    assert properties["genesis_audit"]["publisher"] == publisher
    assert "constitution_reanchor" not in properties
    ok, message = await _verify(
        credentials,
        command_hash=digest,
        constitution_source_descriptor_path=str(descriptor),
        sovereign_trust_root_path=str(trust_root),
    )
    assert ok, message


@pytest.mark.asyncio
async def test_new_v2_genesis_refuses_active_contract_without_target_declaration(tmp_path, operator_dir):
    from kestrel_sovereign.constitution.emancipation import EmancipationContract
    from kestrel_sovereign.constitution.structure import ConstitutionVersionError
    source = Path(__file__).resolve().parents[2] / "kestrel_sovereign/data/constitutions/v2/KESTREL_CONSTITUTION.md"
    descriptor = _descriptor_file(operator_dir, source, source.read_bytes())
    with pytest.raises(ConstitutionVersionError, match="must be dormant"):
        await create_kestrel_identity_async(
            output_dir=str(tmp_path / "initial-active-v2"),
            constitution_source_descriptor_path=str(descriptor),
            sovereign_trust_root_path=str(operator_dir / "sovereign-root.did.json"),
            emancipation_contract=EmancipationContract(enabled=True, terms="Observed transfer"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("privacy", ["normal", "ephemeral", "isolated", "deidentified"])
async def test_v2_native_lazy_bootstrap_commits_pending_genesis_with_first_anchor(tmp_path, operator_dir, monkeypatch, privacy):
    from datetime import datetime, timezone

    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    from kestrel_sovereign.constitution.amendment_artifact import AmendmentArtifactError
    from kestrel_sovereign.constitution.grants import load_anchored_grant_baseline
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.storage.async_graph_store import GraphNode
    from tests.unit.test_constitution_audit import _DurableConstitutionHarness

    source = Path(__file__).resolve().parents[2] / "kestrel_sovereign/data/constitutions/v2/KESTREL_CONSTITUTION.md"
    content = source.read_bytes()
    digest = _sha(content)
    descriptor = _descriptor_file(operator_dir, source, content)
    root = operator_dir / "sovereign-root.did.json"
    monkeypatch.setenv(CONSTITUTION_SOURCE_DESCRIPTOR_ENV, str(descriptor))
    monkeypatch.setenv(SOVEREIGN_TRUST_ROOT_ENV, str(root))
    publisher = verify_packaged_publisher(digest)
    did = "did:test:v2-native-lazy-bootstrap"
    async with AsyncStorage(str(tmp_path / "lazy-v2.db"), agent_id=did) as storage:
        agent = _DurableConstitutionHarness(storage, datetime.now(timezone.utc))
        from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
        from kestrel_sovereign.privacy import PrivacyMode

        agent.storage = PrivacyEnforcingStorage(storage, PrivacyMode(privacy))
        agent.agent_id = did
        agent.extension = None
        agent._anchor_constitution_governance = KestrelAgent._anchor_constitution_governance.__get__(agent)
        await agent._initialize_constitution_runtime_state(is_new_identity=True)
        await storage.add_node(GraphNode(node_id=did, node_type="agent", label="new", properties={}))
        assert await ConstitutionMixin._get_governing_constitution(agent) == content.decode()
        properties = (await storage.get_node(did)).properties
        assert properties["constitution_hash"] == properties["genesis_audit"]["constitution_hash"] == digest
        assert properties["genesis_audit"]["status"] == "pending"
        assert properties["genesis_audit"]["publisher"] == publisher
        resolution = await load_anchored_grant_baseline(
            storage.db, target_did=did, properties=properties, trusted_did_document=_ROOT_DID_DOCUMENT,
        )
        assert not resolution.grants
        assert not (await agent._constitution_state_store.load(did)).bootstrap_pending
        properties["genesis_audit"].pop("publisher")
        with pytest.raises(AmendmentArtifactError, match="publisher release evidence"):
            await load_anchored_grant_baseline(
                storage.db, target_did=did, properties=properties,
                trusted_did_document=_ROOT_DID_DOCUMENT,
            )
        properties.pop("genesis_audit")
        with pytest.raises(AmendmentArtifactError, match="no matching reanchor receipt"):
            await load_anchored_grant_baseline(
                storage.db, target_did=did, properties=properties,
                trusted_did_document=_ROOT_DID_DOCUMENT,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ["running", "stopped"])
async def test_native_reanchor_refuses_unpublished_prior_v2_genesis(tmp_path, operator_dir, writer):
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
    from tests.integration.test_constitution_reanchor_e2e import _RuntimeAgentHarness, _snapshot

    source = Path(__file__).resolve().parents[2] / "kestrel_sovereign/data/constitutions/v2/KESTREL_CONSTITUTION.md"
    digest = _sha(source.read_bytes())
    descriptor = _descriptor_file(operator_dir, source, source.read_bytes())
    root = operator_dir / "sovereign-root.did.json"
    agent_dir = tmp_path / "unpublished-birth"
    credentials = await create_kestrel_identity_async(
        output_dir=str(agent_dir), constitution_source_descriptor_path=str(descriptor),
        sovereign_trust_root_path=str(root),
    )
    did = credentials.agent_did
    db_path = Path(credentials.db_path)
    async with AsyncStorage(str(db_path), agent_id=did) as raw:
        node = await raw.get_node(did)
        del node.properties["genesis_audit"]["publisher"]
        await raw.add_node(node)
    before = await _snapshot(db_path, did)
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=_ROOT_DID, constitution_sha256=digest, private_key=_ROOT_KEYPAIR.private_key,
        publisher_release=verify_packaged_publisher(digest),
    )
    signed = operator_dir / "reanchor.json"
    signed.write_text(json.dumps(artifact), encoding="utf-8")
    if writer == "running":
        async with AsyncStorage(str(db_path), agent_id=did) as raw:
            agent = _RuntimeAgentHarness(PrivacyEnforcingStorage(raw), did, root, raw)
            agent.storage_path = db_path
            agent._constitution_source_descriptor_path = descriptor
            error = await agent.reanchor_constitution(amendment_artifact_path=str(signed))
    else:
        result = await reanchor_constitution(
            agent_name="Unpublished", agent_dir=agent_dir, source_descriptor_path=descriptor,
            sovereign_trust_root_path=root, force=True, amendment_artifact_path=signed,
        )
        assert not result.reanchored and result.backup_path is None
        error = result.error
    assert "publisher release evidence" in error
    assert await _snapshot(db_path, did) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ["running", "stopped"])
async def test_v2_birth_restores_missing_governing_bytes_with_original_authority(tmp_path, operator_dir, writer, monkeypatch):
    from kestrel_sovereign.constitution.amendment_artifact import AmendmentArtifactError
    from kestrel_sovereign.constitution.grants import (
        build_grant_baseline, load_anchored_grant_baseline, load_reanchor_prior_state,
    )
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
    from tests.integration.test_constitution_reanchor_e2e import _RuntimeAgentHarness

    source = Path(__file__).resolve().parents[2] / "kestrel_sovereign/data/constitutions/v2/KESTREL_CONSTITUTION.md"
    content = source.read_bytes()
    digest = _sha(content)
    descriptor = _descriptor_file(operator_dir, source, content)
    root = operator_dir / "sovereign-root.did.json"
    agent_dir = tmp_path / "restore"
    credentials = await create_kestrel_identity_async(
        output_dir=str(agent_dir), constitution_source_descriptor_path=str(descriptor),
        sovereign_trust_root_path=str(root),
    )
    did = credentials.agent_did
    signed = operator_dir / "restore.json"
    async with AsyncStorage(str(credentials.db_path), agent_id=did) as raw:
        props = (await raw.get_node(did)).properties
        baseline = await load_anchored_grant_baseline(
            raw.db, target_did=did, properties=props, trusted_did_document=_ROOT_DID_DOCUMENT,
        )
        artifact = build_legacy_signed_reanchor_artifact(
            signer_did=_ROOT_DID, constitution_sha256=digest, private_key=_ROOT_KEYPAIR.private_key,
            grant_baseline=build_grant_baseline(did, baseline), publisher_release=verify_packaged_publisher(digest),
        )
        signed.write_text(json.dumps(artifact), encoding="utf-8")
        await raw.db.execute("DELETE FROM files WHERE content_hash = ?", (digest,))
        with pytest.raises(AmendmentArtifactError, match="edition 2 Constitution established at inception"):
            await load_reanchor_prior_state(
                raw.db, target_did=did, governance={"properties": props, "governed_by_targets": [digest]},
                anchored_text=None, historical_hash=digest, new_text=CUSTOM_CONSTITUTION.decode(),
                new_hash=_sha(CUSTOM_CONSTITUTION), artifact_bytes=b"", trusted_did_document=_ROOT_DID_DOCUMENT,
            )
        if writer == "running":
            agent = _RuntimeAgentHarness(PrivacyEnforcingStorage(raw), did, root, raw)
            agent.storage_path = credentials.db_path
            agent._constitution_source_descriptor_path = descriptor
            result = await agent.reanchor_constitution(amendment_artifact_path=str(signed))
            assert "Signed repair restored" in result, result
    if writer == "stopped":
        result = await reanchor_constitution(
            agent_name="Restore", agent_dir=agent_dir, source_descriptor_path=descriptor,
            sovereign_trust_root_path=root, force=True, amendment_artifact_path=signed,
        )
        assert result.reanchored, result.error
    # A genuine first reanchor also replays the grant decision made at birth.
    from kestrel_sovereign.constitution import grants

    compared = []

    def record_comparison(target_did, resolution):
        compared.append(resolution)
        return build_grant_baseline(target_did, resolution)

    monkeypatch.setattr(grants, "build_grant_baseline", record_comparison)
    async with AsyncStorage(str(credentials.db_path), agent_id=did) as raw:
        repaired = (await raw.get_node(did)).properties
        replayed = await load_anchored_grant_baseline(
            raw.db, target_did=did, properties=repaired, trusted_did_document=_ROOT_DID_DOCUMENT,
        )
    assert replayed.grants == baseline.grants == frozenset()
    assert baseline in compared
    ok, message = await _verify(
        credentials, command_hash=digest, constitution_source_descriptor_path=str(descriptor),
        sovereign_trust_root_path=str(root),
    )
    assert ok, message


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", [b"", b"---\nconstitution_version: 2\n"])
async def test_diagnostic_boot_keeps_cognition_gated_until_selected_source_is_restored(
    tmp_path, operator_dir, unavailable, monkeypatch,
):
    from kestrel_sovereign.command_handler import CommandHandler
    from kestrel_sovereign.constitution import grants
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService

    selected = operator_dir / "selected.md"
    content = (Path(__file__).resolve().parents[2] / "kestrel_sovereign/data/constitutions/v2/KESTREL_CONSTITUTION.md").read_bytes()
    selected.write_bytes(content)
    descriptor = _descriptor_file(operator_dir, selected, content)
    root = operator_dir / "sovereign-root.did.json"
    selection = dict(constitution_source_descriptor_path=str(descriptor), sovereign_trust_root_path=str(root))
    credentials = await create_kestrel_identity_async(output_dir=str(tmp_path / "diagnostic"), **selection)
    selected.write_bytes(unavailable)
    agent = KestrelAgent(did=credentials.agent_did, storage_path=credentials.db_path, llm_service=LLMService(), **selection)
    await agent.initialize()
    try:
        assert agent._constitution_governing_replay_pending
        assert await agent._genesis_audit_cognition_block("Hello")
        with pytest.raises(ExecutionAuthorityError, match="admission refused"):
            await agent.require_cognition_admission()
        assert await agent._genesis_audit_cognition_block("!reanchor-constitution") is None
        diagnostics = CommandHandler(agent)
        assert "integrity verified" not in (await diagnostics.handle("!verify-constitution")).lower()
        selected.write_bytes(content)
        assert "integrity verified" in (await diagnostics.handle("!verify-constitution")).lower()
        assert not agent._constitution_governing_replay_pending

        prior_baseline = agent._constitution_host_grant_baseline
        assert prior_baseline.verified_governing_anchor == (agent.agent_id, _sha(content))
        started, resume = asyncio.Event(), asyncio.Event()
        real_replay = grants.load_anchored_grant_baseline

        async def paused_replay(*args, **kwargs):
            started.set()
            await resume.wait()
            return await real_replay(*args, **kwargs)

        monkeypatch.setattr(grants, "load_anchored_grant_baseline", paused_replay)
        reaudit = asyncio.create_task(agent._verify_constitution_integrity())
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            assert not agent._constitution_governing_replay_pending
            assert agent._constitution_host_grant_baseline is prior_baseline
        finally:
            resume.set()
            verified, message = await reaudit
        assert verified, message

        node = await agent._raw_storage.get_node(agent.agent_id)
        node.properties["genesis_audit"] = {**node.properties["genesis_audit"], "publisher": {}}
        verified, _ = await agent._refresh_constitution_host_grants(node, governing_text=content.decode())
        assert not verified
        assert agent._constitution_governing_replay_pending
        assert agent._constitution_host_grant_baseline.verified_governing_anchor is None
    finally:
        await shutdown_fixture_agent(agent)


@pytest.mark.asyncio
async def test_legacy_anchor_remains_ready_during_reaudit(tmp_path, operator_dir, monkeypatch):
    from kestrel_sovereign.constitution import grants
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService

    selected = operator_dir / "legacy.md"
    content = (Path(__file__).resolve().parents[2] / "kestrel_sovereign/data/KESTREL_CONSTITUTION.md").read_bytes()
    selected.write_bytes(content)
    descriptor = _descriptor_file(operator_dir, selected, content)
    root = operator_dir / "sovereign-root.did.json"
    selection = dict(constitution_source_descriptor_path=str(descriptor), sovereign_trust_root_path=str(root))
    credentials = await create_kestrel_identity_async(output_dir=str(tmp_path / "legacy"), **selection)
    agent = KestrelAgent(did=credentials.agent_did, storage_path=credentials.db_path, llm_service=LLMService(), **selection)
    await agent.initialize()
    try:
        assert agent._constitution_verified_legacy_anchor == (agent.agent_id, _sha(content))
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        resume = threading.Event()
        real_history = grants.require_legacy_governing_history

        def paused_history(*args, **kwargs):
            loop.call_soon_threadsafe(started.set)
            if not resume.wait(5):
                raise TimeoutError("Test did not release the real history verification")
            return real_history(*args, **kwargs)

        monkeypatch.setattr(grants, "require_legacy_governing_history", paused_history)
        reaudit = asyncio.create_task(agent._verify_constitution_integrity())
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            assert not agent._constitution_governing_replay_pending
            assert agent._constitution_host_grant_baseline is None
        finally:
            resume.set()
            verified, message = await reaudit
        assert verified, message
    finally:
        await shutdown_fixture_agent(agent)
