"""Verify independent v1→v2 adoption authority over the same declaration."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from kestrel_sovereign.config import CONSTITUTION_PATH
from kestrel_sovereign.constitution.amendment_artifact import (
    AmendmentArtifactError,
    build_legacy_signed_reanchor_artifact,
    canonical_amendment_bytes,
    did_document_from_legacy_public_key,
    verify_reanchor_artifact,
)
from kestrel_sovereign.constitution.transition import (
    TRANSITION_PURPOSE,
    build_rotated_transition,
    build_transition,
    verify_transition_adoption,
)
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.rotation_ceremony import run_rotation_ceremony
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from kestrel_sovereign.inception_service import public_key_to_ethereum_address
from kestrel_sovereign.security.crypto_suite import (
    ALG_SLH_DSA_SHA2_128S, Secp256k1Suite, SLHDSASHA2128sSuite,
)

_V1 = Path(CONSTITUTION_PATH).read_text(encoding="utf-8")
_V2 = (
    Path(CONSTITUTION_PATH).parent / "constitutions" / "v2" / "KESTREL_CONSTITUTION.md"
).read_text(encoding="utf-8")
_SUITE = Secp256k1Suite()


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _did(keypair):
    return f"did:pkh:eip155:1:{public_key_to_ethereum_address(keypair.public_key)}"


def test_unsigned_extension_leaves_legacy_canonical_bytes_identical():
    artifact = {
        "artifact_type": "kestrel.constitution.reanchor.v1",
        "version": 1,
        "signer": "did:example:root",
        "subject": "constitution_reanchor",
        "constitution_sha256": "a" * 64,
        "created_at": "2026-11-21T10:00:00+00:00",
    }
    canonical = canonical_amendment_bytes(artifact)
    assert canonical == (
        b'{"artifact_type":"kestrel.constitution.reanchor.v1",'
        b'"constitution_sha256":"' + b'a' * 64 + b'",'
        b'"created_at":"2026-11-21T10:00:00+00:00","reason":"",'
        b'"signer":"did:example:root","subject":"constitution_reanchor","version":1}'
    )
    artifact["transition"] = None
    assert canonical_amendment_bytes(artifact) != canonical


@pytest.fixture
def declaration(tmp_path):
    genesis_key = _SUITE.generate_keypair()
    sovereign = _SUITE.generate_keypair()
    genesis_did = _did(genesis_key)
    sovereign_did = _did(sovereign)
    root = did_document_from_legacy_public_key(sovereign_did, sovereign.public_key)
    intent = {
        "purpose": TRANSITION_PURPOSE,
        "target_did": genesis_did,
        "sovereign_did": sovereign_did,
        "prior_sha256": _hash(_V1),
        "constitution_sha256": _hash(_V2),
        "source_sha256": _hash(_V2),
        "publisher_commit": "1" * 40,
    }
    inputs = dict(
        trusted_did_document=root, target_did=genesis_did, prior_hash=_hash(_V1),
        prior_text=_V1, anchored_contract=None, new_text=_V2, new_hash=_hash(_V2),
        source_hash=_hash(_V2), identity_directory=tmp_path,
    )
    return genesis_key, sovereign, intent, inputs


def test_final_root_signature_covers_genesis_evidence_and_root_is_independent(declaration):
    genesis_key, sovereign, intent, inputs = declaration
    transition = build_transition(intent, genesis_private_key=genesis_key.private_key)
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=_did(sovereign), constitution_sha256=inputs["new_hash"],
        private_key=sovereign.private_key, transition=transition,
    )
    root = inputs["trusted_did_document"]
    assert verify_reanchor_artifact(
        artifact, trusted_did_document=root, expected_constitution_sha256=inputs["new_hash"],
    ).ok
    verify_transition_adoption(artifact=artifact, **inputs)

    artifact["transition"]["intent"]["publisher_commit"] = "2" * 40
    assert not verify_reanchor_artifact(
        artifact, trusted_did_document=root, expected_constitution_sha256=inputs["new_hash"],
    ).ok

    # Re-labeling the Genesis key as a distinct root DID would otherwise
    # let one signature authority occupy both roles.
    relabeled = did_document_from_legacy_public_key(_did(sovereign), genesis_key.public_key)
    with pytest.raises(AmendmentArtifactError, match="cannot also serve"):
        verify_transition_adoption(
            artifact={"transition": build_transition(intent, genesis_private_key=genesis_key.private_key)},
            **(inputs | {"trusted_did_document": relabeled}),
        )


@pytest.mark.parametrize("change", ["prior", "unsupported_chain", "wrong_target", "unknown_v1"])
def test_refuses_wrong_bindings_and_unwitnessed_prior_promises(declaration, change):
    genesis, _sovereign, intent, inputs = declaration
    if change == "prior":
        intent["prior_sha256"] = "0" * 64
    elif change == "unsupported_chain":
        intent["target_did"] = inputs["target_did"] = intent["target_did"].replace(":1:", ":2:")
    elif change == "wrong_target":
        intent["target_did"] = _did(_SUITE.generate_keypair())
    else:
        inputs["prior_text"] = _V1 + "\nThe agent may earn sovereignty under conditions not in a receipt."
        inputs["prior_hash"] = intent["prior_sha256"] = _hash(inputs["prior_text"])
    transition = build_transition(intent, genesis_private_key=genesis.private_key)
    with pytest.raises(AmendmentArtifactError):
        verify_transition_adoption(artifact={"transition": transition}, **inputs)


def test_same_hash_repair_and_v2_reanchor_keep_existing_authority(declaration):
    _genesis, _sovereign, _intent, inputs = declaration
    verify_transition_adoption(
        artifact={}, **(inputs | {"prior_hash": inputs["new_hash"], "prior_text": None}),
    )
    prior = _V2 + "\nPrior edition 2 release notes.\n"
    verify_transition_adoption(
        artifact={}, **(inputs | {"prior_hash": _hash(prior), "prior_text": prior}),
    )


def _rotate(genesis_key, intent, directory, monkeypatch):
    target = intent["target_did"]
    methods = build_verification_methods(target, [(_SUITE, genesis_key.public_key)])
    result = run_rotation_ceremony(
        predecessor_did=target, predecessor_keypair=genesis_key,
        predecessor_kid=methods[0]["id"].rsplit("#", 1)[-1],
        predecessor_verification_methods=methods,
        new_did_domain="agents.test.example", new_did_slug="adopter",
        effective_from="2000-01-01T00:00:00+00:00", reason="isolated adoption test",
        archival_keypair=SLHDSASHA2128sSuite().generate_keypair(),
    )
    published_did_document = deepcopy(result.new_identity.did_document)
    monkeypatch.setattr(
        "kestrel_sovereign.identity.did_web.resolve", lambda did: published_did_document,
    )
    succession_dir = directory / "successions"
    succession_dir.mkdir()
    (succession_dir / "adopter.json").write_text(json.dumps(result.succession_statement.to_dict()))
    identity = AgentIdentity(
        legacy_did=target, new_did=result.new_identity.did,
        hybrid_keypair=result.new_identity.keypair,
        new_verification_methods=list(result.succession_statement.successor_verification_methods),
        succession_chain=result.chain,
    )
    return build_rotated_transition(
        intent, identity=identity, archival_scheme=ALG_SLH_DSA_SHA2_128S,
        archival_multibase=result.succession_statement.archival_verification_method["publicKeyMultibase"],
    )


@pytest.mark.parametrize(
    "change, failure",
    [
        ("valid", None),
        ("wrong_target", "recorded succession"),
        ("chain", "recorded succession"),
        ("terminal_signature", "verification failed"),
        ("missing_algorithm", "both hybrid signature algorithms"),
        ("retired_direct", "recorded succession"),
        ("archival_pin", "archival key"),
        ("missing_publication", "published successor DID"),
    ],
)
def test_rotated_genesis_requires_exact_chain_and_both_active_keys(declaration, change, failure, monkeypatch):
    genesis, _sovereign, intent, inputs = declaration
    transition = deepcopy(_rotate(genesis, intent, inputs["identity_directory"], monkeypatch))
    proof = transition["genesis_proof"]
    if change == "wrong_target":
        proof["trust"]["root_did"] = _did(_SUITE.generate_keypair())
    elif change == "chain":
        proof["trust"]["successions"][0]["reason"] = "unrelated ceremony"
    elif change == "terminal_signature":
        proof["signatures"][0]["sig"] = "00" * 64
    elif change == "missing_algorithm":
        proof["signatures"].pop()
    elif change == "retired_direct":
        transition = build_transition(intent, genesis_private_key=genesis.private_key)
    elif change == "archival_pin":
        proof["archival_multibase"] = "unsupported-key"
    elif change == "missing_publication":
        from kestrel_sovereign.identity.did_web import DidWebError
        def resolve_missing(did):
            raise DidWebError("publication unavailable")
        monkeypatch.setattr("kestrel_sovereign.identity.did_web.resolve", resolve_missing)

    if failure is None:
        verify_transition_adoption(artifact={"transition": transition}, **inputs)
    else:
        with pytest.raises(AmendmentArtifactError, match=failure):
            verify_transition_adoption(artifact={"transition": transition}, **inputs)


def test_historical_rotated_proof_uses_spx_without_relabeling_as_fips(declaration, monkeypatch):
    import os
    import subprocess

    from kestrel_sovereign.constitution.historical_archival import (
        ENV_PYTHON,
        verify_historical_archival_signature,
    )
    from kestrel_sovereign.constitution.transition import HISTORICAL_ARCHIVAL_SCHEME
    from kestrel_sovereign.identity.succession import SuccessionStatement, signable_payload, verify_succession

    interpreter = os.environ.get(ENV_PYTHON)
    if not interpreter:
        pytest.skip("Isolated pqcrypto 0.4 historical verifier is not configured")
    genesis, _sovereign, intent, inputs = declaration
    transition = _rotate(genesis, intent, inputs["identity_directory"], monkeypatch)
    proof = transition["genesis_proof"]
    raw = proof["trust"]["successions"][0]
    statement = SuccessionStatement.from_dict(raw)
    signer = """\
import json
import sys
from pqcrypto.sign import sphincs_sha2_128s_simple as spx

public, private = spx.generate_keypair()
signature = spx.sign(private, sys.stdin.buffer.read())
print(json.dumps({"public": public.hex(), "signature": signature.hex()}))
"""
    result = subprocess.run(
        [interpreter, "-I", "-c", signer],
        input=signable_payload(statement), capture_output=True, check=True, timeout=10, env={},
    )
    archival = json.loads(result.stdout)
    method = build_verification_methods(
        statement.successor_did,
        [(SLHDSASHA2128sSuite(), bytes.fromhex(archival["public"]))],
        kid_prefix="archival",
    )[0]
    raw["archival_verification_method"] = method
    raw["archival_signature"] = {
        "alg": ALG_SLH_DSA_SHA2_128S,
        "kid": method["id"].rsplit("#", 1)[-1],
        "sig": archival["signature"],
    }
    local_record = inputs["identity_directory"] / "successions" / "adopter.json"
    local_record.write_text(json.dumps(raw))
    proof["archival_multibase"] = method["publicKeyMultibase"]

    with pytest.raises(AmendmentArtifactError, match="archival"):
        verify_transition_adoption(artifact={"transition": transition}, **inputs)
    proof["archival_scheme"] = HISTORICAL_ARCHIVAL_SCHEME
    verify_transition_adoption(artifact={"transition": transition}, **inputs)

    from kestrel_sovereign.identity import did_web
    observation = verify_succession(
        SuccessionStatement.from_dict(raw),
        require_archival=True,
        trusted_archival_multibase=proof["archival_multibase"],
        historical_archival_verifier=verify_historical_archival_signature,
        did_web_resolver=did_web.resolve,
    )
    assert observation.ok, observation.reason
    assert observation.archival.alg_ids_seen == {HISTORICAL_ARCHIVAL_SCHEME}

    monkeypatch.delenv(ENV_PYTHON)
    with pytest.raises(AmendmentArtifactError, match=ENV_PYTHON):
        verify_transition_adoption(artifact={"transition": transition}, **inputs)
