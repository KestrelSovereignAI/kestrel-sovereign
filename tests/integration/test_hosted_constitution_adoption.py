"""A directoryless hosted tenant can adopt using independent public custody."""

from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json

import pytest

from kestrel_sovereign.constitution.amendment_artifact import (
    build_legacy_signed_reanchor_artifact,
    did_document_from_legacy_public_key,
)
from kestrel_sovereign.constitution.grants import (
    build_grant_baseline,
    build_offline_grant_observation,
    preview_offline_legacy_grants,
)
from kestrel_sovereign.constitution.publisher import verify_packaged_publisher
from kestrel_sovereign.constitution.source_descriptor import build_legacy_signed_source_descriptor
from kestrel_sovereign.constitution.transition import (
    TRANSITION_PURPOSE,
    build_transition,
    publisher_source_url,
)
from kestrel_sovereign.constitution.workspace_disposition import (
    build_legacy_signed_workspace_disposition,
    configured_workspace_policy,
)
from kestrel_sovereign.identity.signing import get_key_id
from kestrel_sovereign.inception_service import create_kestrel_identity_async, public_key_to_ethereum_address
from kestrel_sovereign.security.crypto_suite import Secp256k1Suite
from kestrel_sovereign.security.key_storage import SecureKeyStorage
from kestrel_sovereign.setup.constitution_reanchor import ReanchorTarget, reanchor_constitution
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.mark.asyncio
@pytest.mark.native_postgres
@pytest.mark.parametrize("db_backend", ["postgres"], indirect=True)
async def test_directoryless_hosted_postgres_adoption(db_backend, tmp_path):
    dsn = db_backend._dsn
    db = await AsyncDatabase.postgres(dsn)
    identity_staging = tmp_path / "inception-only"
    try:
        identity = await create_kestrel_identity_async(
            output_dir=str(identity_staging), database=db, identity_method="did:pkh",
            agent_name="directoryless test tenant", is_test_instance=True,
        )
    finally:
        await db.close()

    target = identity.agent_did
    key_id = get_key_id(target)
    private_key = SecureKeyStorage(identity_staging).load_private_key(key_id)
    public_document = json.loads((identity_staging / f"{key_id}.json").read_text())
    custody = tmp_path / "operator-custody.json"
    custody.write_text(json.dumps({
        "version": 1, "target_did": target, "state": "legacy-only",
        "genesis_document": public_document,
    }))

    sovereign = Secp256k1Suite().generate_keypair()
    signer = "did:pkh:eip155:1:" + public_key_to_ethereum_address(sovereign.public_key)
    root = tmp_path / "operator-root.json"
    root.write_text(json.dumps(did_document_from_legacy_public_key(signer, sovereign.public_key)))
    data = files("kestrel_sovereign").joinpath("data")
    prior = hashlib.sha256(data.joinpath("KESTREL_CONSTITUTION.md").read_bytes()).hexdigest()
    edition_two = data.joinpath("constitutions", "v2", "KESTREL_CONSTITUTION.md").read_bytes()
    digest = hashlib.sha256(edition_two).hexdigest()
    publisher = verify_packaged_publisher(digest)
    source = tmp_path / "published-constitution.md"
    source.write_bytes(edition_two)
    descriptor = tmp_path / "source.json"
    descriptor.write_text(json.dumps(build_legacy_signed_source_descriptor(
        signer_did=signer, source_kind="external", source_path=str(source),
        content_sha256=digest, private_key=sovereign.private_key,
    )))
    env = {"KESTREL_CAPABILITY_GRANTS": "filesystem_read"}
    resolution = preview_offline_legacy_grants(None, overlay_hash=None, environ=env)
    timestamp = datetime.now(timezone.utc).isoformat()
    intent = {
        "purpose": TRANSITION_PURPOSE, "target_did": target, "sovereign_did": signer,
        "prior_sha256": prior, "constitution_sha256": digest, "source_sha256": digest,
        "publisher_commit": publisher["publisher_commit"],
        "publisher_source_url": publisher_source_url(publisher["publisher_commit"]),
        "admission_at": timestamp,
    }
    hosted_config = tmp_path / "tenant-kestrel.toml"
    hosted_config.write_text(
        "[features.computer_use]\nenabled = true\nbackend = 'none'\n"
        "allowed_paths = ['*.md']\nauto_approve_read = false\n"
    )
    policy = configured_workspace_policy(None, hosted_config_path=hosted_config)
    baseline = build_grant_baseline(target, resolution)
    scope = build_legacy_signed_workspace_disposition(
        signer_did=signer, private_key=sovereign.private_key, target_did=target,
        prior_sha256=prior, constitution_sha256=digest, grant_baseline=baseline,
        path_policy=policy["path_policy"],
    )
    artifact = tmp_path / "adoption.json"
    artifact.write_text(json.dumps(build_legacy_signed_reanchor_artifact(
        signer_did=signer, constitution_sha256=digest, private_key=sovereign.private_key,
        transition=build_transition(intent, genesis_private_key=private_key),
        grant_baseline=baseline, publisher_release=publisher, workspace_disposition=scope,
        offline_grant_observation=build_offline_grant_observation(
            target_did=target, prior_sha256=prior, observed_at=timestamp,
            resolution=resolution, workspace_policy=policy,
        ),
    )))
    # This synthetic inception staging is no longer available to the writer.
    identity_staging.rename(tmp_path / "retained-separately")
    args = dict(
        agent_name="directoryless test tenant", agent_dir=None, runtime_backend="postgres",
        runtime_dsn=dsn, hosted_agent_did=target, force=True, canonical_path=source,
        source_descriptor_path=descriptor, sovereign_trust_root_path=root,
        amendment_artifact_path=artifact, environ=env,
    )
    tenant = ReanchorTarget(None, "postgres", target, dsn)
    refused = await reanchor_constitution(**args, hosted_identity_evidence_path=artifact)
    assert not refused.reanchored and "public Genesis evidence" in refused.error
    async with tenant.open_storage(cold_read=True) as stored:
        before = (await stored.graph.get_node(target)).properties
        assert before["constitution_hash"] == prior
        assert "constitution_reanchor" not in before

    hosted_args = {**args, "hosted_identity_evidence_path": custody}
    refused = await reanchor_constitution(**hosted_args)
    assert not refused.reanchored and "operator-selected ComputerUse configuration" in refused.error
    refused = await reanchor_constitution(**hosted_args, hosted_computer_use_config_path=artifact)
    assert not refused.reanchored and "Invalid operator-selected hosted ComputerUse configuration" in refused.error
    original_config = hosted_config.read_text()
    hosted_config.write_text(original_config.replace("enabled = true", "enabled = false"))
    refused = await reanchor_constitution(**hosted_args, hosted_computer_use_config_path=hosted_config)
    assert not refused.reanchored and "differs from the disk-selected policy" in refused.error
    hosted_config.write_text(original_config)
    accepted = await reanchor_constitution(**hosted_args, hosted_computer_use_config_path=hosted_config)
    assert accepted.reanchored, accepted.error
    assert accepted.new_hash == digest
    assert accepted.backup_path is None
    async with tenant.open_storage(cold_read=True) as stored:
        properties = (await stored.graph.get_node(target)).properties
        receipt = properties["constitution_reanchor"]
        assert (receipt["old_hash"], receipt["new_hash"]) == (prior, digest)
        assert properties["genesis_audit"]["status"] == "pending"
        assert await stored.retrieve_file(digest) == edition_two
        assert await stored.retrieve_file(receipt["signed_artifact_hash"]) == artifact.read_bytes()
