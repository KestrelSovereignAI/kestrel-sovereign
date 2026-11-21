"""Separate proof and exact policy matching for the edition-two cutover."""
import hashlib
import json

from cryptography.hazmat.primitives.asymmetric import ec
import pytest

from kestrel_sovereign.constitution.amendment_artifact import (
    AmendmentArtifactError, build_legacy_signed_reanchor_artifact,
    did_document_from_legacy_public_key, verify_reanchor_artifact,
)
from kestrel_sovereign.constitution.grants import GrantResolution, build_grant_baseline, validate_grant_cutover
from kestrel_sovereign.constitution.workspace_disposition import (
    build_legacy_signed_workspace_disposition, workspace_path_policy_evidence,
)
from kestrel_sovereign.features.computer_use.policy import load_computer_use_config

V1 = "# Edition one"
V2 = "---\nconstitution_version: 2\n---\n"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_native_cutover_requires_independent_matching_workspace_signature(tmp_path):
    (tmp_path / "kestrel.toml").write_text(
        '[features.computer_use]\nenabled = true\n'
        f'allowed_paths = [{json.dumps(str(tmp_path / "allow"))}]\n'
        'deny_paths = ["**/restricted/**"]\n'
    )
    target = "did:test:workspace-adopter"
    sovereign = "did:test:workspace-sovereign"
    key = ec.generate_private_key(ec.SECP256K1())
    root = did_document_from_legacy_public_key(sovereign, key.public_key())
    existing = GrantResolution(frozenset({"filesystem_read", "filesystem_write"}), "legacy", "unavailable", "absent")
    baseline = build_grant_baseline(target, existing)
    storage_path = tmp_path / "kestrel_prime.db"
    evidence = workspace_path_policy_evidence(load_computer_use_config(storage_path))
    assert len(evidence["deny_sha256"]) == 1
    assert str(tmp_path) not in json.dumps(evidence)

    def sign(**kwargs):
        return build_legacy_signed_workspace_disposition(
            signer_did=sovereign, private_key=key, target_did=target,
            prior_sha256=kwargs.get("prior", sha(V1)), constitution_sha256=sha(V2),
            grant_baseline=baseline, path_policy=kwargs.get("policy", evidence),
        )

    def authorize(proof=None):
        artifact = build_legacy_signed_reanchor_artifact(
            signer_did=sovereign, private_key=key, constitution_sha256=sha(V2),
            grant_baseline=baseline, workspace_disposition=proof,
        )
        assert verify_reanchor_artifact(artifact, trusted_did_document=root, expected_constitution_sha256=sha(V2)).ok
        return validate_grant_cutover(
            target_did=target, artifact=artifact, prior_text=V1, new_text=V2,
            legacy_resolution=lambda: existing, trusted_did_document=root, storage_path=storage_path,
        )

    with pytest.raises(AmendmentArtifactError, match="separate signed workspace disposition"):
        authorize()
    with pytest.raises(AmendmentArtifactError, match="agent, baseline or governing bytes"):
        authorize(sign(prior="1" * 64))
    with pytest.raises(AmendmentArtifactError, match="differs from the active legacy path policy"):
        authorize(sign(policy={**evidence, "auto_approve_read": False}))
    broken = sign()
    signature = broken["signature"]["sig"]
    broken["signature"]["sig"] = ("0" if signature[0] != "0" else "1") + signature[1:]
    with pytest.raises(AmendmentArtifactError, match="failed signature"):
        authorize(broken)
    accepted = authorize(sign())
    assert accepted.workspace_disposition is not None
    assert accepted.grants == existing.grants


def test_stopped_adoption_requires_signed_observation_when_disabled_on_disk(tmp_path):
    target = "did:test:workspace-adopter"
    resolution = GrantResolution(frozenset({"filesystem_read"}), "legacy", "unavailable", "absent")
    baseline = build_grant_baseline(target, resolution)
    artifact = {"grant_baseline": baseline}
    inputs = dict(
        target_did=target, artifact=artifact, prior_text=V1, new_text=V2,
        legacy_resolution=lambda: resolution, storage_path=tmp_path / "kestrel_prime.db",
    )
    with pytest.raises(AmendmentArtifactError, match="signed pre-stop ComputerUse policy observation"):
        validate_grant_cutover(**inputs, stopped=True)
    artifact["offline_grant_observation"] = {"workspace_policy": {"enabled": False}}
    assert validate_grant_cutover(**inputs, stopped=True).grants == resolution.grants
    with pytest.raises(AmendmentArtifactError, match="Running ComputerUse policy differs"):
        validate_grant_cutover(**inputs, live_workspace_policy=lambda: {"enabled": True})
