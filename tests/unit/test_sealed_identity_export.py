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
