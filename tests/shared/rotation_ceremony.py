"""Reusable, isolated post-ceremony test material.

The archival SLH-DSA signature and encrypted key writes make a complete
rotation ceremony expensive.  Tests that consume the resulting on-disk
shape should not repeat that setup for every assertion, but they also must
not share mutable key directories or shallow-frozen ceremony objects.

This helper builds one real ceremony directory and exposes only per-test
copies plus immutable scalar metadata.  Callers load fresh key/identity
objects from their copy.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.rotation_ceremony import run_rotation_ceremony
from kestrel_sovereign.inception_service import public_key_to_ethereum_address
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    Keypair,
    Secp256k1Suite,
    SLHDSASHA2128sSuite,
)
from kestrel_sovereign.security.key_storage import SecureKeyStorage


TEST_DATA_KEY = "x" * 32
_SLUG = "testbot"


@dataclass(frozen=True)
class PostCeremonyMaterial:
    """An isolated ceremony directory and immutable expected metadata."""

    storage_dir: Path
    legacy_key_id: str
    legacy_did: str
    new_did: str
    slug: str

    def load_legacy_keypair(self) -> Keypair:
        """Load a fresh legacy key object from this test's directory."""
        private_key = SecureKeyStorage(
            storage_dir=self.storage_dir
        ).load_private_key(self.legacy_key_id)
        return Keypair(
            suite_id=ALG_ECDSA_SECP256K1_SHA256,
            private_key=private_key,
            public_key=private_key.public_key(),
        )


@dataclass(frozen=True)
class PostCeremonyTemplate:
    """Private session template; callers receive copies, never this path."""

    source_dir: Path
    legacy_key_id: str
    legacy_did: str
    new_did: str
    slug: str

    def clone_into(self, destination: Path) -> PostCeremonyMaterial:
        shutil.copytree(self.source_dir, destination, dirs_exist_ok=True)
        return PostCeremonyMaterial(
            storage_dir=destination,
            legacy_key_id=self.legacy_key_id,
            legacy_did=self.legacy_did,
            new_did=self.new_did,
            slug=self.slug,
        )


def build_post_ceremony_template(storage_dir: Path) -> PostCeremonyTemplate:
    """Build one real, fully persisted legacy-to-hybrid rotation."""
    storage = SecureKeyStorage(storage_dir=storage_dir)
    secp = Secp256k1Suite()
    legacy_keypair = secp.generate_keypair()
    address = public_key_to_ethereum_address(legacy_keypair.public_key)
    legacy_did = f"did:pkh:eip155:1:{address}"
    legacy_key_id = f"kestrel_{address}"
    storage.save_private_key(legacy_keypair.private_key, legacy_key_id)

    public_key_hex = legacy_keypair.public_key.public_bytes(
        encoding=Encoding.X962,
        format=PublicFormat.UncompressedPoint,
    ).hex()
    did_document = {
        "@context": "https://w3id.org/did/v1",
        "id": legacy_did,
        "publicKey": [
            {
                "id": f"{legacy_did}#keys-1",
                "type": "EcdsaSecp256k1VerificationKey2019",
                "controller": legacy_did,
                "publicKeyHex": public_key_hex,
            }
        ],
    }
    (storage_dir / f"{legacy_key_id}.json").write_text(
        json.dumps(did_document, indent=2),
        encoding="utf-8",
    )

    legacy_verification_methods = build_verification_methods(
        legacy_did,
        [(secp, legacy_keypair.public_key)],
    )
    archival_keypair = SLHDSASHA2128sSuite().generate_keypair()
    result = run_rotation_ceremony(
        predecessor_did=legacy_did,
        predecessor_keypair=legacy_keypair,
        predecessor_kid=legacy_verification_methods[0]["id"].rsplit("#", 1)[-1],
        predecessor_verification_methods=legacy_verification_methods,
        new_did_domain="agents.test.example",
        new_did_slug=_SLUG,
        reason="shared post-ceremony test material",
        effective_from="2000-01-01T00:00:00+00:00",
        archival_keypair=archival_keypair,
    )

    new_keypair = result.new_identity.keypair
    storage.save_private_key(new_keypair.classical.private_key, f"{_SLUG}_ed25519")
    storage.save_secret_bytes(new_keypair.pq.private_key, f"{_SLUG}_mldsa65")
    storage.save_secret_bytes(
        archival_keypair.private_key,
        f"{_SLUG}_archival_slhdsa",
    )
    storage.save_secret_bytes(
        archival_keypair.public_key,
        f"{_SLUG}_archival_slhdsa_pub",
    )
    successions_dir = storage_dir / "successions"
    successions_dir.mkdir()
    (successions_dir / f"{_SLUG}.json").write_text(
        json.dumps(result.succession_statement.to_dict(), indent=2),
        encoding="utf-8",
    )

    return PostCeremonyTemplate(
        source_dir=storage_dir,
        legacy_key_id=legacy_key_id,
        legacy_did=legacy_did,
        new_did=result.new_identity.did,
        slug=_SLUG,
    )
