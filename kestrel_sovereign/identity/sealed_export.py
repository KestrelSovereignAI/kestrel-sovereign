"""
Sealed identity exports — wiring hybrid-KEM sealed capsules into the
identity export/import flow (#2398, real completion of #919 / epic #921).

The generic sealed-capsule primitive (``security.sealed_capsule``:
X25519 + ML-KEM-768 hybrid KEM → HKDF → KSAv2 AEAD) shipped in Wave 4
but wrapped nothing. This module is the integration layer for the
identity-package path — the actual HNDL surface: an identity package
shipped agent-to-agent can be harvested from the transport today and
decrypted in 2040 unless it is sealed for its recipient with a hybrid
KEM.

Three concerns live here:

1. **Recipient resolution** — turning a recipient's published hybrid
   KEM public keys (explicit multibase strings, or ``keyAgreement``
   verification methods in a DID document) into a
   :class:`RecipientKEMKeys` bundle. Missing or invalid keys fail
   loud; there is no plaintext fallback.
2. **Local KEM keypair storage** — an agent that wants to RECEIVE
   sealed exports needs a hybrid KEM keypair of its own, persisted
   the same way the hybrid SIGNING keys are (``SecureKeyStorage``,
   ``<slug>_x25519.key.enc`` + ``<slug>_mlkem768.bytes.enc`` +
   ``<slug>_mlkem768_pub.bytes.enc`` — mirroring the
   ``<slug>_ed25519`` / ``<slug>_mldsa65`` / ``<slug>_archival_slhdsa_pub``
   convention in ``runtime_identity`` / the rotation ceremony).
   The KEM keypair is deliberately SEPARATE from the signing keys:
   the KEM registry and the CryptoSuite registry are split so a
   key-agreement key can never be used as a signing key (or vice
   versa).
3. **Seal / unseal of serialized packages** — a sealed export is the
   package's canonical JSON wrapped in a ``kestrel-sealed-capsule-v1``
   envelope. The importer side detects the envelope by its ``format``
   field and unseals with the local KEM private keys before the
   existing import path runs. Legacy plaintext-JSON exports are
   untouched: :func:`open_identity_export` routes them straight to
   ``AgentIdentityPackage.from_json``.

What this module does NOT cover
-------------------------------

- The CAR / sovereignty export path (``storage/sovereign_adapter``).
  That path is owner-keyed convergent encryption (design principle 5:
  local/at-rest symmetric crypto stays symmetric) with no recipient
  concept — dedup against the owner's own secret, not an
  agent-to-agent shipment. Sealing it is a separate design.
- Sender authentication. Sign the package (``identity.signing``)
  BEFORE sealing; the signature travels inside the capsule.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sovereign.security.hybrid_kem import (
    HybridKEMKeypair,
    generate_hybrid_kem_keypair,
)
from kestrel_sovereign.security.kem_suite import (
    ALG_ML_KEM_768,
    ALG_X25519,
    KEMKeypair,
    KEMSuiteError,
    MLKEM768Suite,
    get_kem_suite,
)
from kestrel_sovereign.security.key_storage import SecureKeyStorage
from kestrel_sovereign.security.multikey import (
    multibase_to_kem_public_key,
    public_key_to_multibase,
)
from kestrel_sovereign.security.sealed_capsule import (
    CAPSULE_FORMAT_ID,
    SealedCapsuleError,
    open_capsule,
    seal_capsule,
)

from .identity_package import AgentIdentityPackage

logger = logging.getLogger(__name__)


class SealedExportError(Exception):
    """Raised on recipient-key resolution failures, missing local KEM
    keys, and unseal failures (wrong recipient / tampered capsule)."""


# ---------------------------------------------------------------------------
# Recipient resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecipientKEMKeys:
    """A recipient's hybrid KEM public keys, resolved and validated.

    ``classical_alg`` / ``pq_alg`` ride along so the seal call binds
    the exact suites the recipient published rather than assuming the
    current defaults — a future ML-KEM-1024 recipient key seals under
    ML-KEM-1024 without changes here.
    """

    classical_public_key: Any
    pq_public_key: Any
    classical_alg: str = ALG_X25519
    pq_alg: str = ALG_ML_KEM_768


def recipient_keys_from_multibase(
    classical_multibase: str,
    pq_multibase: str,
) -> RecipientKEMKeys:
    """Resolve a recipient from two explicit Multikey ``z...`` strings.

    Both strings must decode against the KEM registry (NOT the signing
    registry — a signing pubkey handed to this function fails loud
    rather than silently sealing to a key the recipient can't
    decapsulate with), and the pair must be one classical + one
    post-quantum suite.
    """
    if not classical_multibase or not isinstance(classical_multibase, str):
        raise SealedExportError(
            "recipient classical KEM public key is missing or not a "
            "multibase string; refusing to seal"
        )
    if not pq_multibase or not isinstance(pq_multibase, str):
        raise SealedExportError(
            "recipient post-quantum KEM public key is missing or not a "
            "multibase string; refusing to seal"
        )

    try:
        classical_suite, classical_pub = multibase_to_kem_public_key(
            classical_multibase
        )
    except (KEMSuiteError, ValueError) as e:
        raise SealedExportError(
            f"recipient classical KEM public key is invalid: {e}"
        ) from e
    try:
        pq_suite, pq_pub = multibase_to_kem_public_key(pq_multibase)
    except (KEMSuiteError, ValueError) as e:
        raise SealedExportError(
            f"recipient post-quantum KEM public key is invalid: {e}"
        ) from e

    if classical_suite.is_post_quantum:
        raise SealedExportError(
            f"recipient 'classical' key decodes as post-quantum suite "
            f"{classical_suite.alg_id!r}; the hybrid pair must be one "
            f"classical + one post-quantum key"
        )
    if not pq_suite.is_post_quantum:
        raise SealedExportError(
            f"recipient 'pq' key decodes as classical suite "
            f"{pq_suite.alg_id!r}; the hybrid pair must be one "
            f"classical + one post-quantum key"
        )

    return RecipientKEMKeys(
        classical_public_key=classical_pub,
        pq_public_key=pq_pub,
        classical_alg=classical_suite.alg_id,
        pq_alg=pq_suite.alg_id,
    )


def recipient_keys_from_did_document(did_document: Dict[str, Any]) -> RecipientKEMKeys:
    """Resolve a recipient's hybrid KEM keys from a DID document.

    Per W3C DID Core, key-agreement keys live under the
    ``keyAgreement`` verification relationship — entries are either
    embedded verification-method objects or string references into the
    top-level ``verificationMethod`` array. Both shapes are handled.

    Requires EXACTLY one classical and one post-quantum KEM key among
    the keyAgreement entries. Zero of either fails loud (no plaintext
    fallback); more than one of either is ambiguous and also fails
    loud — pick explicit multibase strings via
    :func:`recipient_keys_from_multibase` in that case.
    """
    if not isinstance(did_document, dict):
        raise SealedExportError(
            f"DID document must be a dict; got {type(did_document).__name__}"
        )

    # Index VMs by absolute id only. Relative keyAgreement references
    # ("#kem-1") resolve strictly against THIS document's ``id`` — never
    # by a global fragment search, which could match an unrelated
    # did:web:other#kem-1 and seal the export to the wrong recipient.
    doc_id = did_document.get("id") or ""
    vm_by_id: Dict[str, Dict[str, Any]] = {}
    for vm in did_document.get("verificationMethod") or []:
        if isinstance(vm, dict) and vm.get("id"):
            vm_id = vm["id"]
            if vm_id in vm_by_id:
                # A repeated verificationMethod.id is a malformed/merged
                # document. Silently keeping the last would let a
                # reference resolve to the wrong key — fail closed.
                raise SealedExportError(
                    f"DID document for {doc_id!r} has duplicate "
                    f"verificationMethod id {vm_id!r}; refusing to "
                    f"resolve recipient keys from an ambiguous document."
                )
            vm_by_id[vm_id] = vm

    key_agreement = did_document.get("keyAgreement") or []
    if not isinstance(key_agreement, list) or not key_agreement:
        raise SealedExportError(
            f"DID document for {did_document.get('id', '<no id>')!r} has "
            f"no keyAgreement verification methods; the recipient has "
            f"not published hybrid KEM keys. Refusing to seal — have "
            f"them run generate_agent_kem_keypair() and publish the "
            f"keys, or pass explicit multibase strings."
        )

    classical: List[tuple] = []  # (alg_id, public_key)
    pq: List[tuple] = []
    for entry in key_agreement:
        if isinstance(entry, str):
            # DID Core resolution. A relative "#fragment" resolves ONLY
            # against this document's id — never against a VM whose raw
            # id happens to be "#fragment" (a malformed/injected VM must
            # not be selectable). An absolute reference is an exact id
            # match.
            if entry.startswith("#"):
                vm = vm_by_id.get(doc_id + entry) if doc_id else None
            else:
                vm = vm_by_id.get(entry)
            if vm is None:
                # A dangling reference is a malformed document; note it
                # but keep scanning — the required-key check below is
                # the authoritative gate.
                logger.warning(f"keyAgreement references unknown VM {entry!r}")
                continue
        elif isinstance(entry, dict):
            vm = entry
        else:
            continue
        mb = vm.get("publicKeyMultibase")
        if not mb:
            continue
        try:
            suite, pub = multibase_to_kem_public_key(mb)
        except (KEMSuiteError, ValueError):
            # Not a registered KEM codec (e.g. a signing key that was
            # misplaced under keyAgreement). Skip; the gate below
            # decides whether we found a usable pair.
            continue
        if suite.is_post_quantum:
            pq.append((suite.alg_id, pub))
        else:
            classical.append((suite.alg_id, pub))

    if len(classical) != 1 or len(pq) != 1:
        raise SealedExportError(
            f"DID document for {did_document.get('id', '<no id>')!r} must "
            f"carry exactly one classical and one post-quantum KEM key "
            f"under keyAgreement; found {len(classical)} classical and "
            f"{len(pq)} post-quantum. Pass explicit multibase strings "
            f"via recipient_keys_from_multibase() to disambiguate."
        )

    return RecipientKEMKeys(
        classical_public_key=classical[0][1],
        pq_public_key=pq[0][1],
        classical_alg=classical[0][0],
        pq_alg=pq[0][0],
    )


# ---------------------------------------------------------------------------
# Local KEM keypair storage (SecureKeyStorage, signing-key conventions)
# ---------------------------------------------------------------------------
#
# File layout per slug (alongside <slug>_ed25519.key.enc etc.):
#
#   <slug>_x25519.key.enc          X25519 private key (cryptography obj, PEM)
#   <slug>_mlkem768.bytes.enc      ML-KEM-768 secret key (2400 raw bytes)
#   <slug>_mlkem768_pub.bytes.enc  ML-KEM-768 public sidecar (1184 raw bytes)
#
# The public sidecar mirrors <slug>_archival_slhdsa_pub: pqcrypto keys
# don't derive public-from-private, and unlike the signing keys there
# is no published DID document to recover the public half from until
# the agent publishes its keyAgreement entries.

def _kem_key_ids(slug: str) -> tuple[str, str, str]:
    return f"{slug}_x25519", f"{slug}_mlkem768", f"{slug}_mlkem768_pub"


def detect_agent_kem_slug(storage_dir: Optional[Path] = None) -> Optional[str]:
    """Discover the local KEM keypair's slug by globbing for the
    classical-half file (``<slug>_x25519.key.enc``), the same
    file-driven approach ``runtime_identity._detect_hybrid_slug`` uses
    for signing keys.

    Returns None if no KEM keys are present. Raises
    :class:`SealedExportError` if more than one slug is present (the
    directory holds an ambiguous set — the caller must pass an explicit
    slug). This avoids deriving the slug from a DID tail, which is wrong
    for multi-segment DIDs (``did:web:host:agent:v1`` → files use
    ``agent_*``, not ``v1_*``) and for legacy did:pkh recipients.
    """
    if storage_dir is not None:
        directory = Path(storage_dir)
    else:
        # Match SecureKeyStorage's default so discovery and loading look
        # in the same place (not cwd).
        from kestrel_sovereign.storage import get_default_agent_data_dir
        directory = Path(get_default_agent_data_dir())
    candidates = sorted(directory.glob("*_x25519.key.enc"))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise SealedExportError(
            f"multiple KEM keypairs in {directory} "
            f"({[c.name for c in candidates]}); pass an explicit slug."
        )
    return candidates[0].name.removesuffix("_x25519.key.enc")


def has_agent_kem_keypair(slug: str, storage_dir: Optional[Path] = None) -> bool:
    """True if all three KEM key files exist for ``slug``."""
    storage = SecureKeyStorage(storage_dir)
    x_id, pq_id, pq_pub_id = _kem_key_ids(slug)
    return (
        storage.has_key(x_id)
        and storage.has_secret_bytes(pq_id)
        and storage.has_secret_bytes(pq_pub_id)
    )


def generate_agent_kem_keypair(
    slug: str,
    storage_dir: Optional[Path] = None,
) -> HybridKEMKeypair:
    """Generate and persist a hybrid KEM keypair for this agent.

    Refuses to overwrite an existing keypair — key rotation is a
    deliberate ceremony (old capsules sealed to the old keys become
    unopenable), not a side effect of re-running generation.

    Requires ``KESTREL_DATA_KEY`` (same at-rest encryption envelope as
    the signing keys).
    """
    if not slug:
        raise SealedExportError("slug must be non-empty")
    # Refuse if ANY component exists — not just a complete set. A
    # partial set (interrupted write, deleted sidecar) still holds
    # recoverable private-key material; overwriting it would make
    # capsules sealed to the old public keys permanently unopenable.
    storage_probe = SecureKeyStorage(storage_dir)
    x_id, pq_id, pq_pub_id = _kem_key_ids(slug)
    present = [
        name for name, exists in (
            (f"{x_id}.key.enc", storage_probe.has_key(x_id)),
            (f"{pq_id}.bytes.enc", storage_probe.has_secret_bytes(pq_id)),
            (f"{pq_pub_id}.bytes.enc", storage_probe.has_secret_bytes(pq_pub_id)),
        ) if exists
    ]
    if present:
        raise SealedExportError(
            f"KEM key material for slug {slug!r} already exists "
            f"({present}); refusing to overwrite. Rotation or repair of "
            f"a partial set requires explicitly removing the old key "
            f"files first (capsules sealed to the old keys become "
            f"unopenable)."
        )

    hybrid = generate_hybrid_kem_keypair()
    storage_probe.save_private_key(hybrid.classical.private_key, x_id)
    storage_probe.save_secret_bytes(hybrid.pq.private_key, pq_id)
    storage_probe.save_secret_bytes(hybrid.pq.public_key, pq_pub_id)
    logger.info(f"Generated hybrid KEM keypair for slug {slug!r}")
    return hybrid


def load_agent_kem_keypair(
    slug: str,
    storage_dir: Optional[Path] = None,
) -> HybridKEMKeypair:
    """Load this agent's hybrid KEM keypair from encrypted storage.

    Fails loud with an actionable message if any of the three files is
    missing — a partially-present keypair means the generation output
    is incomplete and unsealing would fail confusingly downstream.
    """
    storage = SecureKeyStorage(storage_dir)
    x_id, pq_id, pq_pub_id = _kem_key_ids(slug)

    missing = []
    if not storage.has_key(x_id):
        missing.append(f"{x_id}{SecureKeyStorage.ENCRYPTED_EXTENSION}")
    if not storage.has_secret_bytes(pq_id):
        missing.append(f"{pq_id}.bytes.enc")
    if not storage.has_secret_bytes(pq_pub_id):
        missing.append(f"{pq_pub_id}.bytes.enc")
    if missing:
        raise SealedExportError(
            f"no complete hybrid KEM keypair for slug {slug!r} in "
            f"{storage.storage_dir} (missing: {missing}). This agent "
            f"cannot receive sealed identity exports until one exists — "
            f"generate with generate_agent_kem_keypair({slug!r}) and "
            f"publish the public keys."
        )

    x_priv = storage.load_private_key(x_id)
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    if not isinstance(x_priv, X25519PrivateKey):
        raise SealedExportError(
            f"{x_id} is not an X25519 key: {type(x_priv).__name__}"
        )
    classical = KEMKeypair(
        suite_id=ALG_X25519,
        private_key=x_priv,
        public_key=x_priv.public_key(),
    )

    pq_priv = storage.load_secret_bytes(pq_id)
    pq_pub = storage.load_secret_bytes(pq_pub_id)
    if len(pq_priv) != MLKEM768Suite.SECRET_KEY_SIZE:
        raise SealedExportError(
            f"{pq_id} has {len(pq_priv)} bytes; expected "
            f"{MLKEM768Suite.SECRET_KEY_SIZE} (ML-KEM-768 secret key). "
            f"The key file is corrupted or not an ML-KEM-768 key."
        )
    if len(pq_pub) != MLKEM768Suite.PUBLIC_KEY_SIZE:
        raise SealedExportError(
            f"{pq_pub_id} has {len(pq_pub)} bytes; expected "
            f"{MLKEM768Suite.PUBLIC_KEY_SIZE} (ML-KEM-768 public key)."
        )
    pq_kp = KEMKeypair(
        suite_id=ALG_ML_KEM_768,
        private_key=pq_priv,
        public_key=pq_pub,
    )
    return HybridKEMKeypair(classical=classical, pq=pq_kp)


def agent_kem_public_multibases(hybrid: HybridKEMKeypair) -> tuple[str, str]:
    """The (classical, pq) Multikey strings to publish — e.g. as DID
    ``keyAgreement`` verification methods, or handed out-of-band to a
    peer that wants to seal an export to this agent."""
    classical_suite = get_kem_suite(hybrid.classical.suite_id)
    pq_suite = get_kem_suite(hybrid.pq.suite_id)
    return (
        public_key_to_multibase(classical_suite, hybrid.classical.public_key),
        public_key_to_multibase(pq_suite, hybrid.pq.public_key),
    )


# ---------------------------------------------------------------------------
# Seal / unseal
# ---------------------------------------------------------------------------

def seal_identity_package(
    package: AgentIdentityPackage,
    recipient: RecipientKEMKeys,
) -> str:
    """Wrap a package's canonical JSON in a sealed capsule for ``recipient``.

    Sign the package FIRST (``identity.signing.sign_package``) if
    sender authentication is needed — the capsule itself does not
    authenticate the sealer, but signatures inside the payload survive
    the round trip.
    """
    if not isinstance(package, AgentIdentityPackage):
        raise SealedExportError(
            f"expected AgentIdentityPackage; got {type(package).__name__}"
        )
    if not isinstance(recipient, RecipientKEMKeys):
        raise SealedExportError(
            f"expected RecipientKEMKeys; got {type(recipient).__name__}. "
            f"Resolve recipient keys via recipient_keys_from_multibase() "
            f"or recipient_keys_from_did_document()."
        )
    payload = package.to_json().encode("utf-8")
    try:
        return seal_capsule(
            payload,
            recipient_classical_public_key=recipient.classical_public_key,
            recipient_pq_public_key=recipient.pq_public_key,
            classical_alg=recipient.classical_alg,
            pq_alg=recipient.pq_alg,
        )
    except (SealedCapsuleError, KEMSuiteError) as e:
        raise SealedExportError(
            f"failed to seal identity package: {e}"
        ) from e


def _as_text(serialized) -> Optional[str]:
    """Normalize str/bytes input to text. Returns None for anything
    else. Bytes are accepted because ``Path.read_bytes()`` and network
    reads are common capsule sources, and ``json.loads`` would accept
    them too — so the capsule detectors MUST see them or a byte-encoded
    capsule could slip past into the plaintext parser."""
    if isinstance(serialized, str):
        return serialized
    if isinstance(serialized, (bytes, bytearray)):
        try:
            return bytes(serialized).decode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def is_sealed_identity_export(serialized) -> bool:
    """True if ``serialized`` (str or bytes) is a sealed-capsule
    envelope (as opposed to a legacy plaintext identity-package JSON).
    Detection is by the envelope's ``format`` field; non-JSON input
    returns False and is left for the plaintext path to reject."""
    text = _as_text(serialized)
    if text is None:
        return False
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(data, dict) and data.get("format") == CAPSULE_FORMAT_ID


def _looks_like_tampered_capsule(serialized) -> bool:
    """True if ``serialized`` (str or bytes) carries capsule envelope
    fields but does NOT match the exact format id — i.e. a sealed
    capsule whose ``format`` was stripped or altered in transit. Such
    input must be rejected, never downgraded to the plaintext
    identity-package parser (which would silently produce an empty
    package)."""
    text = _as_text(serialized)
    if text is None:
        return False
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, dict) or data.get("format") == CAPSULE_FORMAT_ID:
        return False
    # The capsule envelope's structural fingerprint (see sealed_capsule).
    # EITHER field is enough: a legitimate identity package never carries
    # a top-level "kem" or "ciphertext", so their presence on a
    # non-matching-format object means a stripped/altered capsule — even
    # if the other field was also stripped in the same tamper.
    return "kem" in data or "ciphertext" in data


def unseal_identity_package(
    capsule: str,
    kem_keypair: HybridKEMKeypair,
) -> AgentIdentityPackage:
    """Open a sealed identity export with the local hybrid KEM keypair.

    Wrong recipient, tampered ciphertext, and malformed envelopes all
    surface as :class:`SealedExportError` — fail closed, no partial
    package.
    """
    try:
        payload = open_capsule(capsule, kem_keypair)
    except SealedCapsuleError as e:
        raise SealedExportError(
            f"failed to unseal identity export: {e}. The capsule was "
            f"likely sealed for a different recipient, or has been "
            f"tampered with in transit."
        ) from e
    try:
        package = AgentIdentityPackage.from_json(payload.decode("utf-8"))
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        # e.g. a sealed payload of valid JSON that isn't an object —
        # from_json calls .get on it (codex P2: the error contract is
        # SealedExportError for ALL malformed sealed input, not a
        # leaked AttributeError).
        AttributeError,
    ) as e:
        raise SealedExportError(
            f"capsule unsealed but the payload is not a valid identity "
            f"package: {e}"
        ) from e
    # from_json fills required fields with empty defaults rather than
    # raising, so a garbage object like {} decodes to a package with an
    # empty did. Reject it here — the sealed-export contract is
    # fail-closed, and a downstream import with allow_unsigned=True must
    # never act on a bogus empty agent (codex P2).
    if not isinstance(package.did, str) or not package.did:
        raise SealedExportError(
            f"capsule unsealed but the payload has no valid agent DID "
            f"(got {type(package.did).__name__}); it is not a valid "
            f"identity package (fail-closed)."
        )
    return package


def open_identity_export(
    serialized,
    *,
    kem_keypair: Optional[HybridKEMKeypair] = None,
    slug: Optional[str] = None,
    storage_dir: Optional[Path] = None,
) -> AgentIdentityPackage:
    """Parse an identity export (str or bytes), sealed or plaintext.

    - Sealed capsule → unseal with ``kem_keypair`` if provided, else
      load the local keypair for ``slug`` from ``storage_dir``. A
      sealed capsule with no way to get local KEM keys fails loud.
    - Legacy plaintext JSON → routed straight to
      ``AgentIdentityPackage.from_json`` (path unchanged).
    """
    # Normalize bytes → str at the boundary: json.loads (inside
    # from_json) accepts bytes, so without this a byte-encoded capsule
    # would slip past the detectors into the plaintext parser and
    # deserialize as an empty package (codex round 8, fail-closed).
    text = _as_text(serialized)
    if text is not None:
        serialized = text
    if is_sealed_identity_export(serialized):
        if kem_keypair is None:
            # Prefer an explicit slug; otherwise discover it from the
            # local key files (robust to multi-segment / did:pkh DIDs).
            # Detection uses the default agent data dir when storage_dir
            # is None, mirroring SecureKeyStorage.
            if slug is None:
                slug = detect_agent_kem_slug(storage_dir)
            if slug is None:
                raise SealedExportError(
                    "this is a SEALED identity export but no KEM keys "
                    "were provided and none were found locally (pass "
                    "kem_keypair=, or slug=/storage_dir= to load this "
                    "agent's local keypair). Sealed exports cannot be "
                    "opened without the recipient's private keys."
                )
            kem_keypair = load_agent_kem_keypair(slug, storage_dir)
        return unseal_identity_package(serialized, kem_keypair)
    if _looks_like_tampered_capsule(serialized):
        # A sealed capsule whose format id was stripped/altered in
        # transit. Never downgrade it to the plaintext parser (which
        # would yield an empty package) — fail closed (codex P2).
        raise SealedExportError(
            "input carries sealed-capsule envelope fields (kem + "
            "ciphertext) but not the expected format identifier; the "
            "capsule appears tampered with. Refusing to parse it as a "
            "plaintext identity package."
        )
    return AgentIdentityPackage.from_json(serialized)


__all__ = [
    "RecipientKEMKeys",
    "SealedExportError",
    "agent_kem_public_multibases",
    "generate_agent_kem_keypair",
    "has_agent_kem_keypair",
    "is_sealed_identity_export",
    "load_agent_kem_keypair",
    "open_identity_export",
    "recipient_keys_from_did_document",
    "recipient_keys_from_multibase",
    "seal_identity_package",
    "unseal_identity_package",
]
