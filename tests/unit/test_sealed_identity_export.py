"""
Sealed identity-export tests (#2398 — wiring sealed capsules into the
identity export/import paths; real completion of #919 / epic #921).

Covers:
- Seal → unseal round-trip preserves the exact package
- Legacy plaintext-JSON export path unchanged (open_identity_export
  routes it straight to from_json)
- Wrong recipient fails closed with a clear error
- Tampered capsule fails closed
- Sealed input with no local KEM keys fails loud with an actionable
  message (no plaintext fallback, no silent downgrade)
- Recipient resolution from explicit multibase strings (valid, swapped
  halves, garbage, missing)
- Recipient resolution from DID-document keyAgreement (embedded VMs,
  reference-style entries, missing/ambiguous keys)
- Local KEM keypair storage: SecureKeyStorage file conventions
  (<slug>_x25519.key.enc, <slug>_mlkem768.bytes.enc,
  <slug>_mlkem768_pub.bytes.enc), refuse-overwrite, load-missing
  actionable error
- IdentityExporter.export_sealed / IdentityImporter.import_serialized
  end-to-end on the default SQLite backend
"""
import json

import pytest
import pytest_asyncio

from kestrel_sovereign.identity.identity_package import AgentIdentityPackage
from kestrel_sovereign.identity.sealed_export import (
    RecipientKEMKeys,
    SealedExportError,
    agent_kem_public_multibases,
    generate_agent_kem_keypair,
    has_agent_kem_keypair,
    is_sealed_identity_export,
    load_agent_kem_keypair,
    open_identity_export,
    recipient_keys_from_did_document,
    recipient_keys_from_multibase,
    seal_identity_package,
    unseal_identity_package,
)
from kestrel_sovereign.security.hybrid_kem import generate_hybrid_kem_keypair
from kestrel_sovereign.security.sealed_capsule import CAPSULE_FORMAT_ID


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recipient_kp():
    """The recipient's hybrid KEM keypair (module-scoped: ML-KEM keygen
    is not free)."""
    return generate_hybrid_kem_keypair()


@pytest.fixture(scope="module")
def recipient(recipient_kp) -> RecipientKEMKeys:
    classical_mb, pq_mb = agent_kem_public_multibases(recipient_kp)
    return recipient_keys_from_multibase(classical_mb, pq_mb)


def _package() -> AgentIdentityPackage:
    pkg = AgentIdentityPackage(
        did="did:pkh:eip155:1:0xSealed",
        agent_name="Sealed Agent",
        created_at="2025-01-01T00:00:00Z",
        constitution_hash="",
        constitution_text="",
        episodes=[{"id": "ep1", "title": "First flight", "summary": "s"}],
    )
    pkg.content_hash = pkg.compute_content_hash()
    return pkg


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_seal_unseal_round_trip(recipient_kp, recipient):
    pkg = _package()
    capsule = seal_identity_package(pkg, recipient)

    assert is_sealed_identity_export(capsule)
    env = json.loads(capsule)
    assert env["format"] == CAPSULE_FORMAT_ID

    recovered = unseal_identity_package(capsule, recipient_kp)
    assert recovered.to_dict() == pkg.to_dict()
    assert recovered.verify_content_hash()


def test_open_identity_export_routes_sealed(recipient_kp, recipient):
    pkg = _package()
    capsule = seal_identity_package(pkg, recipient)
    recovered = open_identity_export(capsule, kem_keypair=recipient_kp)
    assert recovered.to_dict() == pkg.to_dict()


def test_plaintext_export_path_unchanged(recipient_kp):
    """A legacy plaintext-JSON export is NOT a sealed capsule and goes
    straight to from_json — with or without KEM keys in hand."""
    pkg = _package()
    serialized = pkg.to_json()
    assert not is_sealed_identity_export(serialized)

    recovered = open_identity_export(serialized)
    assert recovered.to_dict() == pkg.to_dict()

    # Passing keys anyway must not change the plaintext path
    recovered2 = open_identity_export(serialized, kem_keypair=recipient_kp)
    assert recovered2.to_dict() == pkg.to_dict()


# ---------------------------------------------------------------------------
# Fail-closed: wrong recipient / tampering / missing keys
# ---------------------------------------------------------------------------

def test_wrong_recipient_rejected(recipient):
    capsule = seal_identity_package(_package(), recipient)
    attacker = generate_hybrid_kem_keypair()
    with pytest.raises(SealedExportError, match="different recipient"):
        unseal_identity_package(capsule, attacker)


def test_tampered_capsule_rejected(recipient_kp, recipient):
    capsule = seal_identity_package(_package(), recipient)
    env = json.loads(capsule)
    body = list(env["ciphertext"])
    # Flip a character inside the AEAD token body (past the "KSAv2:" tag)
    pos = 10
    body[pos] = "A" if body[pos] != "A" else "B"
    env["ciphertext"] = "".join(body)
    bad = json.dumps(env, separators=(",", ":"))
    with pytest.raises(SealedExportError, match="failed to unseal"):
        unseal_identity_package(bad, recipient_kp)


def test_sealed_input_without_keys_fails_loud(recipient):
    capsule = seal_identity_package(_package(), recipient)
    with pytest.raises(SealedExportError, match="no KEM keys"):
        open_identity_export(capsule)


def test_unsealed_payload_that_is_not_a_package_rejected(recipient_kp):
    """A capsule whose payload isn't an identity package fails with a
    typed error, not a raw json/KeyError leak."""
    from kestrel_sovereign.security.sealed_capsule import seal_capsule
    capsule = seal_capsule(
        b"\xff\xfe not json",
        recipient_classical_public_key=recipient_kp.classical.public_key,
        recipient_pq_public_key=recipient_kp.pq.public_key,
    )
    with pytest.raises(SealedExportError, match="not a valid identity package"):
        unseal_identity_package(capsule, recipient_kp)


# ---------------------------------------------------------------------------
# Recipient resolution — multibase strings
# ---------------------------------------------------------------------------

def test_recipient_from_multibase_valid(recipient_kp):
    classical_mb, pq_mb = agent_kem_public_multibases(recipient_kp)
    r = recipient_keys_from_multibase(classical_mb, pq_mb)
    assert r.classical_alg == "x25519"
    assert r.pq_alg == "ml-kem-768"


def test_recipient_from_multibase_swapped_halves_rejected(recipient_kp):
    classical_mb, pq_mb = agent_kem_public_multibases(recipient_kp)
    with pytest.raises(SealedExportError, match="decodes as post-quantum"):
        recipient_keys_from_multibase(pq_mb, classical_mb)


def test_recipient_from_multibase_missing_rejected():
    with pytest.raises(SealedExportError, match="missing"):
        recipient_keys_from_multibase("", "zWhatever")


def test_recipient_from_multibase_garbage_rejected(recipient_kp):
    _, pq_mb = agent_kem_public_multibases(recipient_kp)
    with pytest.raises(SealedExportError, match="invalid"):
        recipient_keys_from_multibase("znot-a-real-key", pq_mb)


def test_recipient_from_multibase_signing_key_rejected(recipient_kp):
    """A SIGNING pubkey under keyAgreement semantics must not seal:
    the KEM registry doesn't know signing codecs, so resolution fails
    loud instead of sealing to a key nobody can decapsulate with."""
    from kestrel_sovereign.security.crypto_suite import get_suite, ALG_ED25519
    from kestrel_sovereign.security.multikey import public_key_to_multibase
    suite = get_suite(ALG_ED25519)
    kp = suite.generate_keypair()
    signing_mb = public_key_to_multibase(suite, kp.public_key)
    _, pq_mb = agent_kem_public_multibases(recipient_kp)
    with pytest.raises(SealedExportError, match="invalid"):
        recipient_keys_from_multibase(signing_mb, pq_mb)


# ---------------------------------------------------------------------------
# Recipient resolution — DID document keyAgreement
# ---------------------------------------------------------------------------

def _did_doc_with_key_agreement(recipient_kp, *, embedded: bool) -> dict:
    classical_mb, pq_mb = agent_kem_public_multibases(recipient_kp)
    did = "did:web:agents.example.com:sealed"
    vms = [
        {"id": f"{did}#kem-x25519", "type": "Multikey",
         "controller": did, "publicKeyMultibase": classical_mb},
        {"id": f"{did}#kem-mlkem768", "type": "Multikey",
         "controller": did, "publicKeyMultibase": pq_mb},
    ]
    if embedded:
        return {"id": did, "keyAgreement": vms}
    return {
        "id": did,
        "verificationMethod": vms,
        "keyAgreement": [vm["id"] for vm in vms],
    }


def test_recipient_from_did_document_embedded(recipient_kp):
    doc = _did_doc_with_key_agreement(recipient_kp, embedded=True)
    r = recipient_keys_from_did_document(doc)
    assert r.classical_alg == "x25519" and r.pq_alg == "ml-kem-768"
    # And it actually seals/unseals
    capsule = seal_identity_package(_package(), r)
    assert unseal_identity_package(capsule, recipient_kp).did == _package().did


def test_recipient_from_did_document_references(recipient_kp):
    doc = _did_doc_with_key_agreement(recipient_kp, embedded=False)
    r = recipient_keys_from_did_document(doc)
    assert r.classical_alg == "x25519" and r.pq_alg == "ml-kem-768"


def test_did_document_without_key_agreement_rejected():
    with pytest.raises(SealedExportError, match="no keyAgreement"):
        recipient_keys_from_did_document({"id": "did:web:example.com"})


def test_did_document_missing_pq_key_rejected(recipient_kp):
    classical_mb, _ = agent_kem_public_multibases(recipient_kp)
    doc = {
        "id": "did:web:example.com",
        "keyAgreement": [
            {"id": "did:web:example.com#kem", "type": "Multikey",
             "publicKeyMultibase": classical_mb},
        ],
    }
    with pytest.raises(SealedExportError, match="1 classical and 0 post-quantum"):
        recipient_keys_from_did_document(doc)


def test_did_document_ambiguous_classical_keys_rejected(recipient_kp):
    other = generate_hybrid_kem_keypair()
    mb_a, pq_mb = agent_kem_public_multibases(recipient_kp)
    mb_b, _ = agent_kem_public_multibases(other)
    doc = {
        "id": "did:web:example.com",
        "keyAgreement": [
            {"id": "#a", "publicKeyMultibase": mb_a},
            {"id": "#b", "publicKeyMultibase": mb_b},
            {"id": "#c", "publicKeyMultibase": pq_mb},
        ],
    }
    with pytest.raises(SealedExportError, match="2 classical and 1 post-quantum"):
        recipient_keys_from_did_document(doc)


# ---------------------------------------------------------------------------
# Local KEM keypair storage
# ---------------------------------------------------------------------------

@pytest.fixture
def key_env(tmp_path, monkeypatch):
    """KESTREL_DATA_KEY + isolated storage dir for SecureKeyStorage."""
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-sealed-exports")
    return tmp_path


def test_generate_persists_signing_key_style_files(key_env):
    generate_agent_kem_keypair("meridian", storage_dir=key_env)
    assert (key_env / "meridian_x25519.key.enc").exists()
    assert (key_env / "meridian_mlkem768.bytes.enc").exists()
    assert (key_env / "meridian_mlkem768_pub.bytes.enc").exists()
    assert has_agent_kem_keypair("meridian", storage_dir=key_env)


def test_generate_refuses_overwrite(key_env):
    generate_agent_kem_keypair("meridian", storage_dir=key_env)
    with pytest.raises(SealedExportError, match="already exists"):
        generate_agent_kem_keypair("meridian", storage_dir=key_env)


def test_load_round_trips_through_storage(key_env):
    """Keys loaded from encrypted storage open a capsule sealed to the
    multibases published at generation time — the full receive story."""
    generated = generate_agent_kem_keypair("meridian", storage_dir=key_env)
    classical_mb, pq_mb = agent_kem_public_multibases(generated)

    loaded = load_agent_kem_keypair("meridian", storage_dir=key_env)
    r = recipient_keys_from_multibase(classical_mb, pq_mb)
    pkg = _package()
    capsule = seal_identity_package(pkg, r)
    assert unseal_identity_package(capsule, loaded).to_dict() == pkg.to_dict()


def test_load_missing_keypair_actionable_error(key_env):
    with pytest.raises(SealedExportError, match="generate_agent_kem_keypair"):
        load_agent_kem_keypair("nobody", storage_dir=key_env)


def test_open_identity_export_loads_keys_by_slug(key_env):
    generated = generate_agent_kem_keypair("meridian", storage_dir=key_env)
    classical_mb, pq_mb = agent_kem_public_multibases(generated)
    r = recipient_keys_from_multibase(classical_mb, pq_mb)
    pkg = _package()
    capsule = seal_identity_package(pkg, r)
    recovered = open_identity_export(capsule, slug="meridian", storage_dir=key_env)
    assert recovered.to_dict() == pkg.to_dict()


# ---------------------------------------------------------------------------
# Exporter / importer wiring (default SQLite backend)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def sqlite_db(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    db = await AsyncDatabase.sqlite(str(tmp_path / "sealed.db"))
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_export_sealed_and_import_serialized_round_trip(
    sqlite_db, tmp_path, recipient_kp, recipient
):
    """Full path: IdentityExporter.export_sealed → sealed capsule →
    IdentityImporter.import_serialized unseals and imports."""
    from kestrel_sovereign.identity.exporter import IdentityExporter
    from kestrel_sovereign.identity.importer import IdentityImporter

    exporter = IdentityExporter(
        sqlite_db, "did:pkh:eip155:1:0xExporter", agent_data_dir=tmp_path
    )
    capsule = await exporter.export_sealed(recipient)
    assert is_sealed_identity_export(capsule)
    # The package plaintext must not appear in the capsule
    assert "0xExporter" not in capsule

    importer = IdentityImporter(sqlite_db)
    result = await importer.import_serialized(
        capsule,
        kem_keypair=recipient_kp,
        verify_signature=False,
        allow_unsigned=True,
    )
    assert result.success is True
    assert result.agent_id == "did:pkh:eip155:1:0xExporter"


@pytest.mark.asyncio
async def test_import_serialized_plaintext_path_unchanged(sqlite_db):
    from kestrel_sovereign.identity.importer import IdentityImporter
    pkg = _package()
    importer = IdentityImporter(sqlite_db)
    result = await importer.import_serialized(
        pkg.to_json(), verify_signature=False, allow_unsigned=True
    )
    assert result.success is True
    assert result.agent_id == pkg.did


@pytest.mark.asyncio
async def test_import_serialized_wrong_recipient_fails_closed(sqlite_db, recipient):
    """Nothing may be imported from a capsule sealed for someone else."""
    from kestrel_sovereign.identity.importer import IdentityImporter
    capsule = seal_identity_package(_package(), recipient)
    attacker = generate_hybrid_kem_keypair()
    importer = IdentityImporter(sqlite_db)
    with pytest.raises(SealedExportError, match="different recipient"):
        await importer.import_serialized(
            capsule, kem_keypair=attacker,
            verify_signature=False, allow_unsigned=True,
        )
    # Fail-closed: no episodes were written
    row = await sqlite_db.fetchone("SELECT COUNT(*) FROM memory_episodes")
    assert row[0] == 0


@pytest.mark.asyncio
async def test_import_serialized_sealed_without_keys_fails_loud(sqlite_db, recipient):
    from kestrel_sovereign.identity.importer import IdentityImporter
    capsule = seal_identity_package(_package(), recipient)
    importer = IdentityImporter(sqlite_db)
    with pytest.raises(SealedExportError, match="no KEM keys"):
        await importer.import_serialized(
            capsule, verify_signature=False, allow_unsigned=True
        )


def test_unseal_rejects_json_non_object_payload():
    """A capsule whose decrypted payload is valid JSON but not an
    identity-package object must fail closed as SealedExportError,
    not leak an AttributeError (codex P2)."""
    from kestrel_sovereign.security.hybrid_kem import generate_hybrid_kem_keypair
    from kestrel_sovereign.security.sealed_capsule import seal_capsule
    from kestrel_sovereign.identity.sealed_export import (
        SealedExportError, unseal_identity_package,
    )

    kp = generate_hybrid_kem_keypair()
    capsule = seal_capsule(
        b"[]",
        recipient_classical_public_key=kp.classical.public_key,
        recipient_pq_public_key=kp.pq.public_key,
    )
    with pytest.raises(SealedExportError, match="not a valid identity"):
        unseal_identity_package(capsule, kp)


def test_recipient_resolution_handles_relative_key_agreement_refs():
    """DID Core allows keyAgreement to reference VMs by relative
    fragment ("#kem-1") against absolute VM ids (codex round 2)."""
    from kestrel_sovereign.security.hybrid_kem import generate_hybrid_kem_keypair
    from kestrel_sovereign.identity.sealed_export import (
        agent_kem_public_multibases, recipient_keys_from_did_document,
    )

    kp = generate_hybrid_kem_keypair()
    classical_mb, pq_mb = agent_kem_public_multibases(kp)
    did = "did:web:example.com:recip"
    doc = {
        "id": did,
        "verificationMethod": [
            {"id": f"{did}#kem-1", "type": "Multikey",
             "publicKeyMultibase": classical_mb},
            {"id": f"{did}#kem-2", "type": "Multikey",
             "publicKeyMultibase": pq_mb},
        ],
        "keyAgreement": ["#kem-1", "#kem-2"],
    }
    keys = recipient_keys_from_did_document(doc)
    assert keys.classical_alg == "x25519"
    assert keys.pq_alg == "ml-kem-768"


def test_generate_kem_keypair_refuses_partial_existing_set(tmp_path, monkeypatch):
    """Any surviving KEM key component blocks regeneration — a partial
    set still holds recoverable private material (codex round 2)."""
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    from kestrel_sovereign.identity.sealed_export import (
        SealedExportError, generate_agent_kem_keypair,
    )

    generate_agent_kem_keypair("partialbird", tmp_path)
    # Simulate an interrupted state: only the public sidecar removed
    (tmp_path / "partialbird_mlkem768_pub.bytes.enc").unlink()
    with pytest.raises(SealedExportError, match="already exists"):
        generate_agent_kem_keypair("partialbird", tmp_path)
    # The surviving private material was NOT clobbered
    assert (tmp_path / "partialbird_x25519.key.enc").exists()
    assert (tmp_path / "partialbird_mlkem768.bytes.enc").exists()


def test_unseal_rejects_empty_object_payload():
    """A decrypted JSON object that isn't an identity package (e.g. {})
    must fail closed, not decode to an empty-DID package that a
    downstream allow_unsigned import would act on (codex round 3)."""
    from kestrel_sovereign.security.hybrid_kem import generate_hybrid_kem_keypair
    from kestrel_sovereign.security.sealed_capsule import seal_capsule
    from kestrel_sovereign.identity.sealed_export import (
        SealedExportError, unseal_identity_package,
    )

    kp = generate_hybrid_kem_keypair()
    capsule = seal_capsule(
        b"{}",
        recipient_classical_public_key=kp.classical.public_key,
        recipient_pq_public_key=kp.pq.public_key,
    )
    with pytest.raises(SealedExportError, match="no valid agent DID"):
        unseal_identity_package(capsule, kp)


def test_open_export_rejects_format_stripped_capsule():
    """A capsule with its format id removed must NOT downgrade to the
    plaintext parser (which yields an empty package) — codex round 3."""
    import json as _json
    from kestrel_sovereign.security.hybrid_kem import generate_hybrid_kem_keypair
    from kestrel_sovereign.security.sealed_capsule import seal_capsule
    from kestrel_sovereign.identity.sealed_export import (
        SealedExportError, open_identity_export,
    )

    kp = generate_hybrid_kem_keypair()
    capsule = seal_capsule(
        b'{"did": "did:web:x:y", "agent_name": "Y", "created_at": "t", '
        b'"constitution_hash": "h", "constitution_text": "c"}',
        recipient_classical_public_key=kp.classical.public_key,
        recipient_pq_public_key=kp.pq.public_key,
    )
    envelope = _json.loads(capsule)
    del envelope["format"]  # tamper: strip the format id
    with pytest.raises(SealedExportError, match="tampered"):
        open_identity_export(_json.dumps(envelope), kem_keypair=kp)


def test_did_parser_skips_kem_key_agreement_methods():
    """A born-hybrid signing identity that ALSO publishes KEM
    keyAgreement methods must still load: parse_did_document skips KEM
    Multikeys instead of raising on their multicodec (codex round 4)."""
    from kestrel_sovereign.identity.did_web import build_did_document, parse_did_document
    from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair
    from kestrel_sovereign.security.hybrid_kem import generate_hybrid_kem_keypair
    from kestrel_sovereign.security.crypto_suite import get_suite
    from kestrel_sovereign.identity.sealed_export import agent_kem_public_multibases

    sign_kp = generate_hybrid_keypair()
    did = "did:web:example.com:mixed"
    doc = build_did_document(did, sign_kp.public_keys())
    # Append KEM keyAgreement methods (the #2398 publication shape)
    kem = generate_hybrid_kem_keypair()
    c_mb, pq_mb = agent_kem_public_multibases(kem)
    doc["verificationMethod"].append(
        {"id": f"{did}#kem-1", "type": "Multikey", "publicKeyMultibase": c_mb})
    doc["verificationMethod"].append(
        {"id": f"{did}#kem-2", "type": "Multikey", "publicKeyMultibase": pq_mb})
    doc["keyAgreement"] = [f"{did}#kem-1", f"{did}#kem-2"]

    parsed = parse_did_document(doc)  # must not raise
    algs = {suite.alg_id for _kid, suite, _pub in parsed}
    assert "ed25519" in algs and "ml-dsa-65" in algs
    assert "x25519" not in algs and "ml-kem-768" not in algs


def test_colliding_fragment_relative_ref_is_not_hijacked():
    """A relative #frag ref must resolve against THIS doc's id, not a
    colliding fragment on an unrelated VM (codex round 4)."""
    from kestrel_sovereign.security.hybrid_kem import generate_hybrid_kem_keypair
    from kestrel_sovereign.identity.sealed_export import (
        SealedExportError, agent_kem_public_multibases,
        recipient_keys_from_did_document,
    )

    mine = generate_hybrid_kem_keypair()
    other = generate_hybrid_kem_keypair()
    my_c, my_pq = agent_kem_public_multibases(mine)
    other_c, _ = agent_kem_public_multibases(other)
    did = "did:web:example.com:me"
    doc = {
        "id": did,
        "verificationMethod": [
            {"id": f"{did}#kem-1", "type": "Multikey", "publicKeyMultibase": my_c},
            {"id": f"{did}#kem-2", "type": "Multikey", "publicKeyMultibase": my_pq},
            # A colliding fragment on a DIFFERENT did — must not be picked
            {"id": "did:web:evil.example#kem-1", "type": "Multikey",
             "publicKeyMultibase": other_c},
        ],
        "keyAgreement": [f"{did}#kem-1", f"{did}#kem-2"],
    }
    keys = recipient_keys_from_did_document(doc)
    # Resolved against doc id → MY classical key, not evil's
    assert keys.classical_public_key.public_bytes_raw() == \
        mine.classical.public_key.public_bytes_raw()


def test_relative_ref_without_self_vm_fails_closed():
    """A relative '#kem-1' with no <doc-id>#kem-1 VM must fail closed —
    never borrow a foreign VM that merely shares the fragment
    (codex round 5, P1)."""
    from kestrel_sovereign.security.hybrid_kem import generate_hybrid_kem_keypair
    from kestrel_sovereign.identity.sealed_export import (
        SealedExportError, agent_kem_public_multibases,
        recipient_keys_from_did_document,
    )

    other = generate_hybrid_kem_keypair()
    other_c, other_pq = agent_kem_public_multibases(other)
    did = "did:web:example.com:me"
    doc = {
        "id": did,
        "verificationMethod": [
            # Only FOREIGN VMs carry the #kem-* fragments
            {"id": "did:web:evil.example#kem-1", "type": "Multikey",
             "publicKeyMultibase": other_c},
            {"id": "did:web:evil.example#kem-2", "type": "Multikey",
             "publicKeyMultibase": other_pq},
        ],
        "keyAgreement": ["#kem-1", "#kem-2"],
    }
    with pytest.raises(SealedExportError):
        recipient_keys_from_did_document(doc)


def test_unseal_rejects_non_string_did():
    """A sealed payload with a truthy non-string did (e.g. 123) must
    fail closed, not leak a TypeError from downstream import (codex
    round 5, P2)."""
    import json as _json
    from kestrel_sovereign.security.hybrid_kem import generate_hybrid_kem_keypair
    from kestrel_sovereign.security.sealed_capsule import seal_capsule
    from kestrel_sovereign.identity.sealed_export import (
        SealedExportError, unseal_identity_package,
    )

    kp = generate_hybrid_kem_keypair()
    payload = _json.dumps({
        "did": 123, "agent_name": "X", "created_at": "t",
        "constitution_hash": "h", "constitution_text": "c",
    }).encode()
    capsule = seal_capsule(
        payload,
        recipient_classical_public_key=kp.classical.public_key,
        recipient_pq_public_key=kp.pq.public_key,
    )
    with pytest.raises(SealedExportError, match="no valid agent DID"):
        unseal_identity_package(capsule, kp)


@pytest.mark.asyncio
async def test_kem_publishing_agent_still_signs_verifiable_packages(tmp_path, monkeypatch):
    """A born-hybrid agent that ALSO published KEM keyAgreement methods
    in its DID doc must still produce verifiable hybrid identity
    packages — new_verification_methods excludes KEM VMs (codex r6)."""
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    monkeypatch.setenv("KESTREL_DID_WEB_DOMAIN", "agents.kestrel-sovereign.test")
    monkeypatch.delenv("KESTREL_IDENTITY_METHOD", raising=False)
    import json as _json
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.identity.runtime_identity import load_agent_identity
    from kestrel_sovereign.identity.identity_package import AgentIdentityPackage
    from kestrel_sovereign.identity.signing import sign_package, verify_package_signature
    from kestrel_sovereign.identity.sealed_export import (
        generate_agent_kem_keypair, agent_kem_public_multibases,
    )

    creds = await create_kestrel_identity_async(
        str(tmp_path), "docs/principles/KESTREL_CONSTITUTION.md",
        agent_name="Publisher",
    )
    slug = creds.agent_did.rsplit(":", 1)[-1]

    # Publish KEM keyAgreement methods into the on-disk DID document
    kem = generate_agent_kem_keypair(slug, tmp_path)
    c_mb, pq_mb = agent_kem_public_multibases(kem)
    did_path = next(tmp_path.glob("*_did.json"))
    doc = _json.loads(did_path.read_text())
    doc["verificationMethod"].append(
        {"id": f"{creds.agent_did}#kem-1", "type": "Multikey", "publicKeyMultibase": c_mb})
    doc["verificationMethod"].append(
        {"id": f"{creds.agent_did}#kem-2", "type": "Multikey", "publicKeyMultibase": pq_mb})
    doc["keyAgreement"] = [f"{creds.agent_did}#kem-1", f"{creds.agent_did}#kem-2"]
    did_path.write_text(_json.dumps(doc))

    ident = load_agent_identity(None, storage_dir=tmp_path)
    # Only signing VMs are exposed for signing
    assert len(ident.new_verification_methods) == 2
    algs = set()
    from kestrel_sovereign.security.multikey import multibase_to_public_key
    for vm in ident.new_verification_methods:
        suite, _ = multibase_to_public_key(vm["publicKeyMultibase"])
        algs.add(suite.alg_id)
    assert algs == {"ed25519", "ml-dsa-65"}

    package = AgentIdentityPackage(
        did=creds.agent_did, agent_name="Publisher",
        created_at="2026-07-11T00:00:00Z",
        constitution_hash="h", constitution_text="c",
    )
    signed = sign_package(package, storage_dir=tmp_path)
    ok, msg = verify_package_signature(signed, storage_dir=tmp_path)
    assert ok, msg


def test_malformed_relative_vm_id_not_selectable():
    """A VM whose raw id is the relative '#kem-1' must not satisfy a
    '#kem-1' keyAgreement ref — only <doc-id>#kem-1 resolves (r7)."""
    from kestrel_sovereign.security.hybrid_kem import generate_hybrid_kem_keypair
    from kestrel_sovereign.identity.sealed_export import (
        SealedExportError, agent_kem_public_multibases,
        recipient_keys_from_did_document,
    )
    evil = generate_hybrid_kem_keypair()
    ec, epq = agent_kem_public_multibases(evil)
    did = "did:web:example.com:me"
    doc = {
        "id": did,
        "verificationMethod": [
            {"id": "#kem-1", "type": "Multikey", "publicKeyMultibase": ec},
            {"id": "#kem-2", "type": "Multikey", "publicKeyMultibase": epq},
        ],
        "keyAgreement": ["#kem-1", "#kem-2"],
    }
    with pytest.raises(SealedExportError):
        recipient_keys_from_did_document(doc)


def test_partial_capsule_fingerprint_rejected():
    """A capsule stripped of format AND ciphertext (only 'kem' left)
    must still fail closed, not parse as an empty package (r7)."""
    from kestrel_sovereign.identity.sealed_export import (
        SealedExportError, open_identity_export,
    )
    import json as _json
    tampered = _json.dumps({"kem": {"whatever": 1}})
    with pytest.raises(SealedExportError, match="tampered"):
        open_identity_export(tampered)
