"""
Runtime identity loader — Wave 3 follow-up of Quantum Hardening (#921).

The Wave 2/3 ceremony tooling produces hybrid keys + succession statements
on disk, but ``inception_service.load_kestrel_identity`` only knew how
to load a single legacy ECDSA private key. This module adds the missing
piece: a runtime-side loader that detects a post-ceremony agent and
returns a richer :class:`AgentIdentity` carrying both the legacy key
(still useful for verifying pre-cutoff artifacts) and the new
:class:`HybridKeypair` (used to sign new artifacts).

Design notes
------------

- **Backward compatible.** The old ``load_kestrel_identity`` function is
  unchanged and still returns ``(legacy_priv, legacy_did_doc)``. Callers
  that need hybrid awareness use :func:`load_agent_identity` instead.
- **State invariants enforced.** If the agent dir has a succession
  statement, the corresponding hybrid keys MUST also exist. The loader
  raises rather than silently falling back to the legacy key (that would
  hide a partial-ceremony failure).
- **Single succession only for now.** Today's deployment has exactly one
  succession per agent (legacy → hybrid). Multi-succession chains
  (e.g. hybrid → re-rotated hybrid) are a future extension; this loader
  picks the lone ``successions/*.json`` entry and assembles the chain
  from it.
- **Slug derivation.** The succession statement carries the new DID
  (e.g. ``did:web:agents.kestrelsovereign.com:kestrel``). The slug after
  the last ``:`` is the prefix used for the hybrid key files
  (``kestrel_ed25519.key.enc`` etc).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric import ec

from kestrel_sovereign.identity.hybrid_keypair import HybridKeypair
from kestrel_sovereign.identity.succession import SuccessionStatement
from kestrel_sovereign.identity.succession_chain import (
    SuccessionChain,
    build_chain,
)
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    ALG_ED25519,
    ALG_ML_DSA_65,
    ALG_SLH_DSA_SHA2_128S,
    Keypair,
)
from kestrel_sovereign.security.key_storage import SecureKeyStorage


logger = logging.getLogger(__name__)


class RuntimeIdentityError(Exception):
    """Raised on inconsistent on-disk identity state (e.g. succession
    statement exists but the hybrid key files don't, or vice versa)."""


@dataclass(frozen=True)
class AgentIdentity:
    """Runtime view of an agent's identity.

    Carries enough state for the agent to:

    - Sign new artifacts (use ``hybrid_keypair`` if ``is_hybrid``,
      otherwise the legacy keypair).
    - Verify older artifacts under the chain walker (use
      ``succession_chain`` if ``is_hybrid``).
    - Identify itself when publishing (``signing_did``).
    """

    # Always present (every agent has a legacy identity)
    legacy_did: str
    legacy_keypair: Keypair
    legacy_did_document: dict

    # Set iff the agent's data dir contains a succession statement
    hybrid_keypair: Optional[HybridKeypair] = None
    new_did: Optional[str] = None
    # Verification methods (Multikey VMs) for the new DID, copied
    # verbatim from the succession statement. We deliberately do NOT
    # reconstruct a full DID document here: the published one at
    # ``did:web``'s HTTPS URL may carry alsoKnownAs / service / context
    # entries that the runtime can't reproduce from the statement
    # alone. Callers that need the full document fetch it from the
    # network or from the agent-identities repo. Callers that need to
    # know which kid maps to which alg use this field directly.
    new_verification_methods: Optional[list] = None
    succession_chain: Optional[SuccessionChain] = None
    archival_keypair: Optional[Keypair] = None
    succession_statement: Optional[SuccessionStatement] = None

    @property
    def is_hybrid(self) -> bool:
        return self.hybrid_keypair is not None

    @property
    def signing_did(self) -> str:
        """The DID the agent should sign new artifacts AS, right now.

        Post-ceremony, that's the new ``did:web`` identity. Before any
        rotation, it's the legacy ``did:pkh``.
        """
        return self.new_did if self.is_hybrid else self.legacy_did


def _detect_hybrid_slug(storage_dir: Path) -> str:
    """Find the slug used for hybrid key files in this agent dir.

    The ceremony writes ``<slug>_ed25519.key.enc`` (along with
    ``<slug>_mldsa65.bytes.enc`` and the archival pair). We discover
    the slug by globbing for the classical-half file rather than by
    parsing the new DID URI — that way we don't break on multi-segment
    DIDs like ``did:web:domain:agent:v1`` where ``rsplit(':')[-1]``
    would return ``v1`` instead of the actual key-file prefix.
    """
    candidates = sorted(storage_dir.glob("*_ed25519.key.enc"))
    if not candidates:
        raise RuntimeIdentityError(
            f"no hybrid classical key (*_ed25519.key.enc) in {storage_dir}; "
            f"succession statement is present but the ceremony output "
            f"is incomplete."
        )
    if len(candidates) > 1:
        raise RuntimeIdentityError(
            f"multiple hybrid classical keys in {storage_dir}: "
            f"{[c.name for c in candidates]}. Slug is ambiguous."
        )
    return candidates[0].name.removesuffix("_ed25519.key.enc")


def _load_legacy_part(
    storage: SecureKeyStorage,
    storage_dir: Path,
    legacy_key_id: str,
    *,
    allow_missing_private: bool = False,
) -> tuple[Keypair, dict, str]:
    """Load the legacy ECDSA keypair + DID document. Returns (kp, doc, did).

    With ``allow_missing_private=True``, tolerates a missing legacy
    private key by deriving the public key from the DID document.
    This is ONLY safe in the post-destruction hybrid state: a
    legacy-only agent that lost its private key has no signing
    capability and should fail loud rather than silently loading.
    Callers must only pass ``allow_missing_private=True`` after
    confirming a succession statement is on disk.
    """
    priv = None
    if storage.has_key(legacy_key_id):
        priv = storage.load_private_key(legacy_key_id)
    else:
        # Plaintext PEM fallback — same logic as the existing
        # load_kestrel_identity, kept here so this function is
        # standalone-callable in tests.
        pem_path = storage_dir / f"{legacy_key_id}.pem"
        if pem_path.exists():
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            from cryptography.hazmat.backends import default_backend
            with open(pem_path, "rb") as f:
                priv = load_pem_private_key(
                    f.read(), password=None, backend=default_backend(),
                )
            logger.warning(
                f"Loaded PLAINTEXT legacy key from {pem_path}. "
                "Encrypt at rest with KESTREL_DATA_KEY when convenient."
            )
        elif not allow_missing_private:
            # Legacy-only agent missing its private key has no signing
            # capability and should fail loud — not silently fall
            # through to a public-only state. The post-destruction
            # case is handled at the caller, after a succession
            # statement is confirmed on disk.
            raise FileNotFoundError(
                f"No legacy key for {legacy_key_id} in {storage_dir} "
                f"(checked .key.enc and .pem) and no succession statement "
                f"to justify a public-only fall-through."
            )
        # else: post-destruction hybrid state. Fall through; we
        # build the keypair from the DID document's public-only data.
    if priv is not None and not isinstance(priv, ec.EllipticCurvePrivateKey):
        raise RuntimeIdentityError(
            f"legacy key is not an ECDSA key: {type(priv).__name__}"
        )

    did_path = storage_dir / f"{legacy_key_id}.json"
    if not did_path.exists():
        raise FileNotFoundError(
            f"DID document not found at {did_path}. Cannot load identity "
            f"with neither private key nor DID document."
        )
    did_document = json.loads(did_path.read_text())
    legacy_did = did_document.get("id")
    if not legacy_did:
        raise RuntimeIdentityError(
            f"DID document at {did_path} has no 'id' field"
        )

    if priv is not None:
        pub = priv.public_key()
    else:
        # Post-destruction: derive public from the DID document.
        # The legacy DID document carries publicKeyHex in the
        # ``publicKey``/``verificationMethod`` array.
        pub_hex = None
        for vm in (
            did_document.get("publicKey") or did_document.get("verificationMethod") or []
        ):
            pub_hex = vm.get("publicKeyHex")
            if pub_hex:
                break
        if not pub_hex:
            raise FileNotFoundError(
                f"Legacy private key not on disk for {legacy_key_id} "
                f"AND DID document at {did_path} has no publicKeyHex. "
                f"Cannot reconstruct legacy public key."
            )
        try:
            pub = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256K1(), bytes.fromhex(pub_hex),
            )
        except Exception as e:
            raise RuntimeIdentityError(
                f"Failed to decode legacy public key from DID document: {e}"
            )
        logger.info(
            f"Legacy private key absent (post-destruction) for {legacy_did}; "
            f"derived legacy public from DID document. Hybrid identity "
            f"continues to provide signing capability."
        )

    legacy_kp = Keypair(
        suite_id=ALG_ECDSA_SECP256K1_SHA256,
        private_key=priv,  # may be None post-destruction
        public_key=pub,
    )
    return legacy_kp, did_document, legacy_did


def _find_succession_statement(storage_dir: Path) -> Optional[Path]:
    """Look for a succession statement under <storage_dir>/successions/.

    Returns the path to the statement, or None if no successions/ dir
    exists or it's empty. Raises if multiple statements exist (we don't
    yet support multi-succession runtime identity — pick the chain
    explicitly when that ships).
    """
    successions_dir = storage_dir / "successions"
    if not successions_dir.is_dir():
        return None
    candidates = sorted(successions_dir.glob("*.json"))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise RuntimeIdentityError(
            f"multiple succession statements in {successions_dir}: "
            f"{[c.name for c in candidates]}. Multi-succession runtime "
            f"identity is not yet supported; resolve to a single chain "
            f"and re-archive."
        )
    return candidates[0]


def _load_hybrid_part(
    storage: SecureKeyStorage,
    storage_dir: Path,
    statement: SuccessionStatement,
) -> tuple[HybridKeypair, list, Keypair]:
    """Load the hybrid keypair + verification methods + archival keypair
    pinned by ``statement``. Returns (hybrid_kp, verification_methods,
    archival_kp)."""
    slug = _detect_hybrid_slug(storage_dir)

    # 1) Hybrid classical (Ed25519, PEM-encoded)
    classical_key_id = f"{slug}_ed25519"
    if not storage.has_key(classical_key_id):
        raise RuntimeIdentityError(
            f"succession statement references successor {statement.successor_did!r} "
            f"but classical hybrid key {classical_key_id}.key.enc is missing in "
            f"{storage_dir}. The ceremony output is incomplete; refusing to load."
        )
    ed_priv = storage.load_private_key(classical_key_id)
    if not isinstance(ed_priv, Ed25519PrivateKey):
        raise RuntimeIdentityError(
            f"{classical_key_id}.key.enc is not an Ed25519 key: "
            f"{type(ed_priv).__name__}"
        )
    classical_kp = Keypair(
        suite_id=ALG_ED25519,
        private_key=ed_priv,
        public_key=ed_priv.public_key(),
    )

    # 2) Hybrid PQ (ML-DSA-65, raw bytes)
    pq_key_id = f"{slug}_mldsa65"
    if not storage.has_secret_bytes(pq_key_id):
        raise RuntimeIdentityError(
            f"succession statement references successor {statement.successor_did!r} "
            f"but post-quantum hybrid key {pq_key_id}.bytes.enc is missing in "
            f"{storage_dir}."
        )
    pq_priv_bytes = storage.load_secret_bytes(pq_key_id)
    # Public key for ML-DSA-65 is recovered from the published DID document
    # (we don't store it separately — it's published in the verification methods).
    pq_kp_partial_public = None  # filled in below from the new DID doc
    # 3) New DID document — published locally as part of the ceremony.
    # The successor verification methods on the statement carry the
    # multibase-encoded pubkeys, which we decode to get the raw bytes.
    from kestrel_sovereign.security.multikey import multibase_to_public_key
    pq_pub_bytes = None
    for vm in statement.successor_verification_methods:
        mb = vm.get("publicKeyMultibase")
        if not mb:
            continue
        try:
            suite, pub = multibase_to_public_key(mb)
        except Exception:
            continue
        if suite.alg_id == ALG_ML_DSA_65:
            pq_pub_bytes = pub
            break
    if pq_pub_bytes is None:
        raise RuntimeIdentityError(
            "succession statement carries no ML-DSA-65 verification "
            "method; cannot recover hybrid PQ public key."
        )
    pq_kp = Keypair(
        suite_id=ALG_ML_DSA_65,
        private_key=pq_priv_bytes,
        public_key=pq_pub_bytes,
    )
    hybrid = HybridKeypair(classical=classical_kp, pq=pq_kp)

    # 4) Archival SLH-DSA keypair (private + public sidecar)
    archival_priv_id = f"{slug}_archival_slhdsa"
    archival_pub_id = f"{slug}_archival_slhdsa_pub"
    if not storage.has_secret_bytes(archival_priv_id):
        raise RuntimeIdentityError(
            f"archival SLH-DSA private key {archival_priv_id}.bytes.enc missing "
            f"in {storage_dir} (succession statement carries an archival "
            f"signature, so the key was minted; restore from backup)."
        )
    if not storage.has_secret_bytes(archival_pub_id):
        raise RuntimeIdentityError(
            f"archival SLH-DSA public sidecar {archival_pub_id}.bytes.enc missing "
            f"in {storage_dir}."
        )
    archival_kp = Keypair(
        suite_id=ALG_SLH_DSA_SHA2_128S,
        private_key=storage.load_secret_bytes(archival_priv_id),
        public_key=storage.load_secret_bytes(archival_pub_id),
    )

    # Expose just the verification methods, not a full DID document.
    # The published did.json may carry alsoKnownAs / service / context
    # fields the statement doesn't capture — reproducing a "DID
    # document" here would silently drift from what consumers fetch
    # over HTTPS. The runtime only needs the VMs (kid -> alg mapping)
    # to sign new artifacts.
    new_verification_methods = list(statement.successor_verification_methods)
    return hybrid, new_verification_methods, archival_kp


def load_agent_identity(
    legacy_key_id: str,
    storage_dir: Optional[Path] = None,
) -> AgentIdentity:
    """Load an agent's full runtime identity from disk.

    Always loads the legacy ECDSA keypair + DID document (every Kestrel
    agent has those). If a succession statement exists in
    ``<storage_dir>/successions/``, also loads the corresponding hybrid
    + archival keypairs and exposes them on the returned
    :class:`AgentIdentity`.

    Args:
        legacy_key_id: ``kestrel_<eth_address>`` — the legacy key id.
        storage_dir: Agent data dir. Defaults to
            ``get_default_agent_data_dir()``.

    Returns:
        :class:`AgentIdentity`. ``is_hybrid`` is ``True`` if the agent has
        completed a rotation ceremony.

    Raises:
        FileNotFoundError: legacy key + DID doc not on disk.
        RuntimeIdentityError: succession state is partial / inconsistent
            (e.g. statement present but hybrid keys missing).
    """
    if storage_dir is None:
        from kestrel_sovereign.storage import get_default_agent_data_dir
        storage_dir = Path(get_default_agent_data_dir())
    else:
        storage_dir = Path(storage_dir)

    storage = SecureKeyStorage(storage_dir=storage_dir)
    # Look for a succession statement BEFORE loading the legacy
    # keypair so we can decide whether a missing legacy private key
    # is a recoverable post-destruction state (succession exists =
    # hybrid signing covers us) or a hard failure (legacy-only agent
    # missing its only signing key).
    succession_path = _find_succession_statement(storage_dir)
    legacy_kp, legacy_did_doc, legacy_did = _load_legacy_part(
        storage, storage_dir, legacy_key_id,
        allow_missing_private=(succession_path is not None),
    )

    if succession_path is None:
        return AgentIdentity(
            legacy_did=legacy_did,
            legacy_keypair=legacy_kp,
            legacy_did_document=legacy_did_doc,
        )

    statement = SuccessionStatement.from_dict(
        json.loads(succession_path.read_text()),
    )
    # Sanity: the statement must succeed THIS legacy DID
    if statement.predecessor_did != legacy_did:
        raise RuntimeIdentityError(
            f"succession statement at {succession_path} has predecessor "
            f"{statement.predecessor_did!r} but the loaded legacy DID is "
            f"{legacy_did!r}. Wrong agent dir, or stale succession statement."
        )

    hybrid_kp, new_vms, archival_kp = _load_hybrid_part(
        storage, storage_dir, statement,
    )
    chain = build_chain([statement])

    logger.info(
        f"Loaded hybrid agent identity: legacy={legacy_did} "
        f"-> new={statement.successor_did}"
    )
    return AgentIdentity(
        legacy_did=legacy_did,
        legacy_keypair=legacy_kp,
        legacy_did_document=legacy_did_doc,
        hybrid_keypair=hybrid_kp,
        new_did=statement.successor_did,
        new_verification_methods=new_vms,
        succession_chain=chain,
        archival_keypair=archival_kp,
        succession_statement=statement,
    )


__all__ = [
    "AgentIdentity",
    "RuntimeIdentityError",
    "load_agent_identity",
]
