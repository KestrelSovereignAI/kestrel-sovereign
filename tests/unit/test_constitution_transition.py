"""Verify independent v1→v2 adoption authority over the same declaration."""

from copy import deepcopy
from datetime import datetime, timezone
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
    build_born_hybrid_transition,
    build_rotated_transition,
    build_transition,
    publisher_source_url,
    verify_archived_transition_adoption,
    verify_live_genesis_proof,
    verify_transition_adoption,
)
from kestrel_sovereign.identity.did_web import build_did_document, build_verification_methods
from kestrel_sovereign.identity.rotation_ceremony import run_rotation_ceremony
from kestrel_sovereign.constitution.publisher import verify_packaged_publisher
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair
from kestrel_sovereign.inception_service import public_key_to_ethereum_address
from kestrel_sovereign.security.key_storage import SecureKeyStorage
from kestrel_sovereign.security.crypto_suite import (
    ALG_SLH_DSA_SHA2_128S, Secp256k1Suite, SLHDSASHA2128sSuite, get_suite,
)

_V1 = Path(CONSTITUTION_PATH).read_text(encoding="utf-8")
_V2 = (
    Path(CONSTITUTION_PATH).parent / "constitutions" / "v2" / "KESTREL_CONSTITUTION.md"
).read_text(encoding="utf-8")
_SUITE = Secp256k1Suite()


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _publisher():
    return verify_packaged_publisher(_hash(_V2))


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


def _seed_legacy_custody(directory, keypair):
    from kestrel_sovereign.identity.signing import get_key_id

    did = _did(keypair)
    document = did_document_from_legacy_public_key(did, keypair.public_key)
    key_id = get_key_id(did)
    (directory / f"{key_id}.json").write_text(json.dumps(document))
    SecureKeyStorage(directory, master_key="synthetic-genesis-custody").save_private_key(
        keypair.private_key, key_id,
    )
    return document


@pytest.fixture
def declaration(tmp_path):
    genesis_key = _SUITE.generate_keypair()
    sovereign = _SUITE.generate_keypair()
    genesis_did = _did(genesis_key)
    _seed_legacy_custody(tmp_path, genesis_key)
    sovereign_did = _did(sovereign)
    root = did_document_from_legacy_public_key(sovereign_did, sovereign.public_key)
    intent = {
        "purpose": TRANSITION_PURPOSE,
        "target_did": genesis_did,
        "sovereign_did": sovereign_did,
        "prior_sha256": _hash(_V1),
        "constitution_sha256": _hash(_V2),
        "source_sha256": _hash(_V2),
        "publisher_commit": _publisher()["publisher_commit"],
        "publisher_source_url": publisher_source_url(_publisher()["publisher_commit"]),
        "admission_at": datetime.now(timezone.utc).isoformat(),
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
        private_key=sovereign.private_key, transition=transition, publisher_release=_publisher(),
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
            artifact={"created_at": datetime.now(timezone.utc).isoformat(), "publisher_release": _publisher(), "transition": build_transition(intent, genesis_private_key=genesis_key.private_key)},
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
        verify_transition_adoption(artifact={"created_at": datetime.now(timezone.utc).isoformat(), "publisher_release": _publisher(), "transition": transition}, **inputs)


def test_same_hash_repair_and_v2_reanchor_keep_existing_authority(declaration):
    _genesis, _sovereign, _intent, inputs = declaration
    prior = _V2 + "\nPrior edition 2 release notes.\n"
    signed = {"publisher_release": _publisher(), "created_at": datetime.now(timezone.utc).isoformat()}
    for prior_digest, prior_text in ((inputs["new_hash"], None), (_hash(prior), prior)):
        revision = inputs | {"prior_hash": prior_digest, "prior_text": prior_text}
        with pytest.raises(AmendmentArtifactError, match="publisher-backed governing anchor"):
            verify_transition_adoption(artifact=signed, **revision)
        revision["verified_prior_anchor"] = (inputs["target_did"], prior_digest)
        verify_transition_adoption(artifact=signed, **revision)
        with pytest.raises(AmendmentArtifactError, match="predates"):
            verify_transition_adoption(
                artifact=signed | {"created_at": "2000-01-01T00:00:00+00:00"}, **revision,
            )


@pytest.fixture
def proof_declaration(tmp_path):
    genesis = _SUITE.generate_keypair()
    sovereign = _SUITE.generate_keypair()
    root = did_document_from_legacy_public_key(_did(sovereign), sovereign.public_key)
    target = _did(genesis)
    _seed_legacy_custody(tmp_path, genesis)
    return genesis, sovereign, {
        "target_did": target,
        "admission_at": "2026-11-20T00:00:00+00:00",
    }, {"target_did": target, "trusted_did_document": root, "identity_directory": tmp_path}


def check_live_proof(transition, inputs):
    return verify_live_genesis_proof(
        inputs["target_did"], transition["intent"], transition["genesis_proof"],
        inputs["trusted_did_document"], inputs["identity_directory"],
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
    published_raw = json.dumps(published_did_document)
    monkeypatch.setattr(
        "kestrel_sovereign.identity.did_web.resolve_document",
        lambda did: (published_raw.encode(), published_did_document),
    )
    monkeypatch.setattr(
        "kestrel_sovereign.identity.did_web.resolve", lambda did: published_did_document,
    )
    succession_dir = directory / "successions"
    succession_dir.mkdir()
    (succession_dir / "adopter.json").write_text(json.dumps(result.succession_statement.to_dict()))
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-transition-public-sidecar-custody")
    SecureKeyStorage(directory).save_secret_bytes(
        result.archival_keypair.public_key, "adopter_archival_slhdsa_pub",
    )
    identity = AgentIdentity(
        legacy_did=target, new_did=result.new_identity.did,
        hybrid_keypair=result.new_identity.keypair,
        new_verification_methods=list(result.succession_statement.successor_verification_methods),
        succession_chain=result.chain,
    )
    return build_rotated_transition(
        intent, identity=identity, archival_scheme=ALG_SLH_DSA_SHA2_128S,
        published_did_document=published_raw,
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
        ("archival_pin", "archival pin differs"),
        ("missing_sidecar", "archival public key custody"),
        ("substituted_sidecar", "archival pin differs"),
        ("missing_publication", "published successor DID"),
        ("publication_changed", "differs from the signed snapshot"),
        ("before_rotation", "verification failed"),
    ],
)
def test_rotated_genesis_requires_exact_chain_and_both_active_keys(proof_declaration, change, failure, monkeypatch):
    genesis, _sovereign, intent, inputs = proof_declaration
    if change == "before_rotation":
        intent["admission_at"] = "1999-01-01T00:00:00+00:00"
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
    elif change == "missing_sidecar":
        (inputs["identity_directory"] / "adopter_archival_slhdsa_pub.bytes.enc").unlink()
    elif change == "substituted_sidecar":
        SecureKeyStorage(inputs["identity_directory"]).save_secret_bytes(
            SLHDSASHA2128sSuite().generate_keypair().public_key, "adopter_archival_slhdsa_pub",
        )
    elif change == "missing_publication":
        from kestrel_sovereign.identity.did_web import DidWebError
        def resolve_missing(did):
            raise DidWebError("publication unavailable")
        monkeypatch.setattr("kestrel_sovereign.identity.did_web.resolve_document", resolve_missing)

    elif change == "publication_changed":
        substituted = proof["published_did_document"] + " "
        monkeypatch.setattr(
            "kestrel_sovereign.identity.did_web.resolve_document",
            lambda did: (substituted.encode(), json.loads(substituted)),
        )

    if failure is None:
        check_live_proof(transition, inputs)
    else:
        with pytest.raises(AmendmentArtifactError, match=failure):
            check_live_proof(transition, inputs)


def test_historical_rotated_proof_uses_spx_without_relabeling_as_fips(proof_declaration, monkeypatch):
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
    genesis, _sovereign, intent, inputs = proof_declaration
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
    SecureKeyStorage(inputs["identity_directory"]).save_secret_bytes(
        bytes.fromhex(archival["public"]), "adopter_archival_slhdsa_pub",
    )

    with pytest.raises(AmendmentArtifactError, match="archival"):
        check_live_proof(transition, inputs)
    proof["archival_scheme"] = HISTORICAL_ARCHIVAL_SCHEME
    check_live_proof(transition, inputs)

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
        check_live_proof(transition, inputs)


@pytest.mark.parametrize("rotated", [False, True], ids=["direct-then-rotated", "archived-rotation"])
def test_archived_receipt_replays_without_current_network_or_directory(declaration, monkeypatch, rotated):
    genesis, sovereign, intent, inputs = declaration
    directory = inputs["identity_directory"]
    transition = (
        _rotate(genesis, intent, directory, monkeypatch) if rotated
        else build_transition(intent, genesis_private_key=genesis.private_key)
    )
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=_did(sovereign), constitution_sha256=inputs["new_hash"],
        private_key=sovereign.private_key, transition=transition, publisher_release=_publisher(),
    )
    custody_evidence = verify_transition_adoption(artifact=artifact, **inputs)
    artifact_bytes = json.dumps(artifact).encode()
    receipt = {
        "old_hash": inputs["prior_hash"],
        "new_hash": inputs["new_hash"],
        "source_content_sha256": inputs["source_hash"],
        "signed_artifact_hash": hashlib.sha256(artifact_bytes).hexdigest(),
        "signed_artifact_signer": _did(sovereign),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **custody_evidence,
    }
    if not rotated:
        # A later recorded rotation makes a raw Genesis proof inadmissible for
        # a new write, while the original completed receipt retains its value.
        _rotate(genesis, intent, directory, monkeypatch)
        with pytest.raises(AmendmentArtifactError, match="recorded succession"):
            verify_transition_adoption(artifact=artifact, **inputs)
    (directory / "successions").rename(directory / "unavailable")

    def network_forbidden(*args, **kwargs):
        pytest.fail("Archive verification consulted the live publication")

    monkeypatch.setattr("kestrel_sovereign.identity.did_web.resolve_document", network_forbidden)
    replay_inputs = {key: value for key, value in inputs.items() if key != "identity_directory"}
    verify_archived_transition_adoption(artifact_bytes=artifact_bytes, receipt=receipt, **replay_inputs)
    if rotated:
        with pytest.raises(AmendmentArtifactError, match="recorded admission custody"):
            verify_archived_transition_adoption(
                artifact_bytes=artifact_bytes, receipt=receipt | {"genesis_admission_evidence": {}},
                **replay_inputs,
            )
    with pytest.raises(AmendmentArtifactError, match="postdates"):
        verify_archived_transition_adoption(
            artifact_bytes=artifact_bytes,
            receipt=receipt | {"timestamp": "2001-01-01T00:00:00+00:00"},
            **replay_inputs,
        )
    with pytest.raises(AmendmentArtifactError, match="native receipt"):
        verify_archived_transition_adoption(
            artifact_bytes=artifact_bytes + b" ", receipt=receipt, **replay_inputs,
        )
    bad = json.loads(artifact_bytes)
    bad["transition"]["intent"]["admission_at"] = "2026-02-01T00:00:00+00:00"
    tampered = json.dumps(bad).encode()
    with pytest.raises(AmendmentArtifactError, match="Archived Sovereign declaration failed"):
        verify_archived_transition_adoption(
            artifact_bytes=tampered,
            receipt=receipt | {"signed_artifact_hash": hashlib.sha256(tampered).hexdigest()},
            **replay_inputs,
        )


def test_historical_callback_requires_literal_true_and_reports_unavailability(declaration, monkeypatch):
    from kestrel_sovereign.constitution.historical_archival import HistoricalVerifierError
    from kestrel_sovereign.identity import did_web
    from kestrel_sovereign.identity.succession import SuccessionStatement, verify_succession

    genesis, _sovereign, intent, inputs = declaration
    proof = _rotate(genesis, intent, inputs["identity_directory"], monkeypatch)["genesis_proof"]
    statement = SuccessionStatement.from_dict(proof["trust"]["successions"][0])

    def inspect(callback):
        return verify_succession(
            statement, require_archival=True, trusted_archival_multibase=proof["archival_multibase"],
            historical_archival_verifier=callback, did_web_resolver=did_web.resolve,
        )

    malformed = inspect(lambda *args, **kwargs: (False, "invalid"))
    assert malformed.ok is False and not malformed.archival.ok

    def unavailable(*args, **kwargs):
        raise HistoricalVerifierError("archival backend unavailable")

    failed = inspect(unavailable)
    assert failed.ok is False and "verifier unavailable" in failed.archival.reason


def _born_transition(intent, inputs, monkeypatch, change=None):
    did = "did:web:agents.test.example:born-hybrid"
    intent["target_did"] = inputs["target_did"] = did

    def document_for(hybrid):
        return build_did_document(did, [
            (get_suite(pair.suite_id), pair.public_key) for pair in (hybrid.classical, hybrid.pq)
        ])

    hybrid = generate_hybrid_keypair()
    document = document_for(hybrid)
    published = json.dumps(document)
    local = inputs["identity_directory"] / "born-hybrid_did.json"
    SecureKeyStorage(
        inputs["identity_directory"], master_key="test-born-hybrid-key-custody",
    ).save_private_key(hybrid.classical.private_key, "born-hybrid_ed25519")
    if change == "foreign_custody":
        local.write_text(json.dumps(document_for(generate_hybrid_keypair())))
    elif change != "missing_custody":
        local.write_text(published)
    monkeypatch.setattr(
        "kestrel_sovereign.identity.did_web.resolve_document", lambda did: (published.encode(), document),
    )
    identity = AgentIdentity(
        new_did=did, hybrid_keypair=hybrid, new_verification_methods=document["verificationMethod"],
    )
    transition = build_born_hybrid_transition(intent, identity=identity, published_did_document=published)
    if change == "missing_signature":
        transition["genesis_proof"]["signatures"].pop()
    return transition, local


@pytest.mark.parametrize("change", [None, "foreign_custody", "missing_custody", "missing_signature"])
def test_born_hybrid_admission_requires_independent_custody_and_both_signatures(
    proof_declaration, monkeypatch, change,
):
    _genesis, _sovereign, intent, inputs = proof_declaration
    transition, _local = _born_transition(intent, inputs, monkeypatch, change)
    if change is None:
        assert check_live_proof(transition, inputs)["genesis_admission_evidence"]["published_document_sha256"]
    else:
        with pytest.raises(AmendmentArtifactError, match="Genesis"):
            check_live_proof(transition, inputs)


def test_born_hybrid_adoption_replays_without_current_custody(declaration, monkeypatch):
    _genesis, sovereign, intent, inputs = declaration
    transition, local = _born_transition(intent, inputs, monkeypatch)
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=_did(sovereign), constitution_sha256=inputs["new_hash"],
        private_key=sovereign.private_key, transition=transition, publisher_release=_publisher(),
    )
    evidence = verify_transition_adoption(artifact=artifact, **inputs)
    local.unlink()
    monkeypatch.setattr(
        "kestrel_sovereign.identity.did_web.resolve_document",
        lambda did: pytest.fail("Archive verification consulted current HTTPS"),
    )
    raw = json.dumps(artifact).encode()
    receipt = {
        "old_hash": inputs["prior_hash"],
        "new_hash": inputs["new_hash"],
        "source_content_sha256": inputs["source_hash"],
        "signed_artifact_hash": hashlib.sha256(raw).hexdigest(),
        "signed_artifact_signer": _did(sovereign),
        "timestamp": artifact["created_at"],
        **evidence,
    }
    replay_inputs = {key: value for key, value in inputs.items() if key != "identity_directory"}
    verify_archived_transition_adoption(artifact_bytes=raw, receipt=receipt, **replay_inputs)


@pytest.mark.parametrize("misordered_event", ["publication", "outer_signature", "future_signature"])
def test_signed_adoption_requires_the_actual_order_of_events(declaration, misordered_event):
    from datetime import timedelta

    genesis, sovereign, intent, inputs = declaration
    release = _publisher()
    admitted = datetime.now(timezone.utc)
    signed = admitted
    if misordered_event == "publication":
        admitted = datetime.fromisoformat(release["issued_at"]) - timedelta(seconds=1)
    elif misordered_event == "outer_signature":
        signed = admitted - timedelta(seconds=1)
    else:
        signed = admitted + timedelta(days=1)
    intent = dict(intent, admission_at=admitted.isoformat())
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=_did(sovereign), constitution_sha256=inputs["new_hash"],
        private_key=sovereign.private_key, created_at=signed.isoformat(),
        transition=build_transition(intent, genesis_private_key=genesis.private_key),
        publisher_release=release,
    )
    with pytest.raises(AmendmentArtifactError, match="predates|postdates"):
        verify_transition_adoption(artifact=artifact, **inputs)


@pytest.mark.parametrize("indicator", [None, "archival-sidecar", "active-successor", "missing-legacy"])
def test_direct_genesis_refuses_incomplete_or_conflicting_native_custody(proof_declaration, indicator):
    genesis, _sovereign, intent, inputs = proof_declaration
    directory = inputs["identity_directory"]
    target = inputs["target_did"]
    kwargs = {}
    if indicator == "archival-sidecar":
        (directory / "rotation_archival_slhdsa_pub.bytes.enc").touch()
    elif indicator == "active-successor":
        kwargs["active_signing_did"] = "did:web:agents.test.example:successor"
    elif indicator == "missing-legacy":
        next(directory.glob("kestrel_*.key.enc")).unlink()
    proof = build_transition(intent, genesis_private_key=genesis.private_key)["genesis_proof"]
    args = (target, intent, proof, inputs["trusted_did_document"], directory)
    if indicator is None:
        assert verify_live_genesis_proof(*args) == {}
    else:
        with pytest.raises(AmendmentArtifactError, match="custody|public succession"):
            verify_live_genesis_proof(*args, **kwargs)


def test_directoryless_hosted_adoption_uses_only_independent_public_custody(declaration):
    genesis, sovereign, intent, inputs = declaration
    directory = inputs.pop("identity_directory")
    target = inputs["target_did"]
    path = directory / "operator-public-custody.json"
    custody = {
        "version": 1, "target_did": target, "state": "legacy-only",
        "genesis_document": did_document_from_legacy_public_key(target, genesis.public_key),
    }
    path.write_text(json.dumps(custody))
    for private in directory.glob("*.key.enc"):
        private.unlink()
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=_did(sovereign), constitution_sha256=inputs["new_hash"],
        private_key=sovereign.private_key,
        transition=build_transition(intent, genesis_private_key=genesis.private_key),
        publisher_release=_publisher(),
    )
    kwargs = dict(artifact=artifact, identity_directory=None, **inputs)
    with pytest.raises(AmendmentArtifactError, match="exactly one"):
        verify_transition_adoption(**kwargs)
    assert verify_transition_adoption(**kwargs, hosted_identity_evidence_path=path) == {}
    path.chmod(0o666)
    with pytest.raises(AmendmentArtifactError, match="operator-owned regular file"):
        verify_transition_adoption(**kwargs, hosted_identity_evidence_path=path)
    path.chmod(0o600)
    path.write_text(json.dumps(custody | {"genesis_document": artifact}))
    with pytest.raises(AmendmentArtifactError, match="independently retained"):
        verify_transition_adoption(**kwargs, hosted_identity_evidence_path=path)


def test_live_and_archived_adoption_agree_on_a_newly_activated_contract(declaration, monkeypatch):
    from kestrel_sovereign.constitution.emancipation import EmancipationContract, apply_emancipation

    genesis, sovereign, intent, inputs = declaration
    contract = EmancipationContract(enabled=True, terms="Complete the agreed transfer rehearsal.")
    active = apply_emancipation(_V2, contract)
    digest = _hash(active)
    intent = intent | {"constitution_sha256": digest}
    inputs = inputs | {"new_text": active, "new_hash": digest}
    replay = {key: value for key, value in inputs.items() if key != "identity_directory"}
    activation = {
        "purpose": "kestrel.emancipation.activate.v2", "target_did": inputs["target_did"],
        "prior_sha256": inputs["prior_hash"], "constitution_sha256": digest,
    }
    for proof in (None, activation):
        artifact = build_legacy_signed_reanchor_artifact(
            signer_did=_did(sovereign), private_key=sovereign.private_key, constitution_sha256=digest,
            transition=build_transition(intent, genesis_private_key=genesis.private_key), publisher_release=_publisher(),
            emancipation_activation=proof,
        )
        raw = json.dumps(artifact).encode()
        receipt = {
            "old_hash": inputs["prior_hash"], "new_hash": digest, "source_content_sha256": inputs["source_hash"],
            "signed_artifact_hash": hashlib.sha256(raw).hexdigest(), "signed_artifact_signer": _did(sovereign),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if proof is None:
            with pytest.raises(AmendmentArtifactError, match="separate Sovereign authorization"):
                verify_archived_transition_adoption(artifact_bytes=raw, receipt=receipt, **replay)
            with pytest.raises(AmendmentArtifactError, match="separate Sovereign authorization"):
                verify_transition_adoption(artifact=artifact, proposed_contract=contract, **inputs)
        else:
            verify_transition_adoption(artifact=artifact, proposed_contract=contract, **inputs)
            verify_archived_transition_adoption(artifact_bytes=raw, receipt=receipt, resulting_contract=contract, **replay)

            # Both callers must compare the governing bytes with what the
            # publisher verifier actually read from the archive.
            from kestrel_sovereign.constitution import publisher

            read_members = publisher._verify_members

            def different_source(*args):
                read_members(*args)
                return (_V2 + "\n").encode()

            monkeypatch.setattr(publisher, "_verify_members", different_source)
            with pytest.raises(AmendmentArtifactError, match="governing bytes differ"):
                verify_transition_adoption(artifact=artifact, proposed_contract=contract, **inputs)
            with pytest.raises(AmendmentArtifactError, match="governing bytes differ"):
                verify_archived_transition_adoption(artifact_bytes=raw, receipt=receipt, resulting_contract=contract, **replay)
