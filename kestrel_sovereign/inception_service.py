#!/usr/bin/env python3
"""
Inception Service: A library for programmatically creating new Kestrel agents.

.. note::
   This module must **not** call ``load_dotenv()`` at import time. Doing so
   loaded the *current-directory* ``.env`` (e.g. a source checkout's key) into
   ``os.environ`` the moment the module was imported — even transitively, long
   before any target home was chosen. When setup then generated a *different*
   ``KESTREL_DATA_KEY`` for an explicit ``KESTREL_HOME`` target, inception
   encrypted the born identity with the stale current-directory key while the
   target ``.env`` persisted the freshly-generated one, so an immediate restart
   could not decrypt the identity (issue #2468). Dotenv loading must be
   *target-aware* and is the caller's responsibility. The setup ``keys`` step
   resolves the effective ``KESTREL_DATA_KEY`` deliberately; the CLI
   ``create`` / ``setup agent`` paths resolve it the same way before inception
   (``_apply_target_data_key_custody`` in ``cli.py``), loading the resolved
   project home's ``.env`` and refusing an exported⇄persisted key conflict.
"""

import logging
import json
from pathlib import Path
from dataclasses import dataclass
from cryptography.hazmat.primitives.asymmetric import ec
from kestrel_sovereign.storage import GraphNode
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_file_store import AsyncFileStore
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
from kestrel_sovereign.security.key_storage import secure_delete
import argparse
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kestrel_sovereign.constitution.emancipation import EmancipationContract
    from kestrel_sovereign.constitution.genesis_audit import GenesisAuditor
from datetime import datetime, timezone
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
import asyncio
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AgentCredentials:
    agent_did: str
    db_path: str
    agent_name: str
    backup_prompt: str
    is_test_instance: bool = False
    test_cycle_id: Optional[str] = None
    openrouter_key_hash: Optional[str] = None  # For LLM billing/usage tracking
    is_demo: bool = False  # #766: agent is demo-scoped — destructive ops on it are safe

    # Allow this object to be awaited in async tests while remaining usable in sync code
    def __await__(self):
        async def _return_self():
            return self
        return _return_self().__await__()


def generate_secp256k1_keypair() -> tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
    """Generate a secp256k1 key pair.

    Wave 1 sub-PR 5: routed through ``KeypairFactory`` so future
    hybrid-identity work (Wave 2) flips the default suite without
    touching this function. Behavior is byte-identical to the
    previous direct ``ec.generate_private_key(ec.SECP256K1())`` call —
    pinned by the CryptoSuite behavior-preservation pair test in
    ``tests/unit/test_crypto_suite.py``.
    """
    from kestrel_sovereign.security.crypto_suite import ALG_ECDSA_SECP256K1_SHA256
    from kestrel_sovereign.security.keypair_factory import KeypairFactory

    keypair = KeypairFactory.generate(ALG_ECDSA_SECP256K1_SHA256)
    return keypair.private_key, keypair.public_key


def public_key_to_hex(public_key: ec.EllipticCurvePublicKey) -> str:
    """Convert public key to uncompressed hex format (04 prefix + x + y coordinates)"""
    public_bytes = public_key.public_bytes(
        encoding=Encoding.X962,
        format=PublicFormat.UncompressedPoint
    )
    return public_bytes.hex()


def public_key_to_ethereum_address(public_key: ec.EllipticCurvePublicKey) -> str:
    """Derive Ethereum address from public key using Keccak-256"""
    from Crypto.Hash import keccak

    public_bytes = public_key.public_bytes(
        encoding=Encoding.X962,
        format=PublicFormat.UncompressedPoint
    )

    public_key_bytes = public_bytes[1:]

    keccak_hash = keccak.new(digest_bits=256)
    keccak_hash.update(public_key_bytes)
    hash_bytes = keccak_hash.digest()

    address = "0x" + hash_bytes[-20:].hex()

    return apply_checksum(address)


def apply_checksum(address: str) -> str:
    """Apply EIP-55 checksum to Ethereum address"""
    from Crypto.Hash import keccak

    address = address[2:].lower()

    keccak_hash = keccak.new(digest_bits=256)
    keccak_hash.update(address.encode('utf-8'))
    hash_hex = keccak_hash.hexdigest()

    checksum_address = "0x"
    for i, char in enumerate(address):
        if char.isalpha():
            checksum_address += char.upper() if int(hash_hex[i], 16) >= 8 else char.lower()
        else:
            checksum_address += char

    return checksum_address


def create_did_document(public_key_hex: str, ethereum_address: str) -> dict:
    """Create a W3C DID document for the Kestrel agent"""
    did_id = f"did:pkh:eip155:1:{ethereum_address}"

    return {
        "@context": "https://w3id.org/did/v1",
        "id": did_id,
        "alsoKnownAs": ["Kestrel"],
        "publicKey": [{
            "id": f"{did_id}#keys-1",
            "type": "EcdsaSecp256k1VerificationKey2019",
            "controller": did_id,
            "publicKeyHex": public_key_hex
        }],
        "authentication": [f"{did_id}#keys-1"],
        "assertionMethod": [f"{did_id}#keys-1"],
        "created": datetime.now(timezone.utc).isoformat(),
        "updated": datetime.now(timezone.utc).isoformat(),
        "note": "Kestrel agent identity - cryptographic proof of continuity"
    }


def save_kestrel_identity(did_document: dict, keys: dict, key_id: str, output_dir: Path):
    """Saves the agent's identity and private key securely.

    The private key is encrypted at rest using KESTREL_DATA_KEY.
    Falls back to plaintext PEM if KESTREL_DATA_KEY is not set (with warning).

    Args:
        did_document: The DID document to save
        keys: Dictionary containing 'private_key_obj', 'public_key_hex', 'address'
        key_id: Identifier for the key (typically kestrel_{address})
        output_dir: Directory to save files to
    """
    private_key = keys['private_key_obj']
    output_dir = Path(output_dir)

    # Try to use secure encrypted storage
    try:
        from kestrel_sovereign.security.key_storage import SecureKeyStorage, MasterKeyNotConfiguredError
        storage = SecureKeyStorage(storage_dir=output_dir)
        storage.save_private_key(private_key, key_id)
        logging.info(f"Saved encrypted private key for {key_id}")
    except MasterKeyNotConfiguredError:
        # Fall back to plaintext with a warning
        logging.warning(
            "SECURITY WARNING: KESTREL_DATA_KEY not set. "
            "Private key will be saved in PLAINTEXT. "
            "Set KESTREL_DATA_KEY for encrypted key storage."
        )
        # Save plaintext as before
        pem_path = output_dir / f"{key_id}.pem"
        private_pem = private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption()
        )
        with open(pem_path, 'wb') as f:
            f.write(private_pem)
        os.chmod(pem_path, 0o600)
        logging.info(f"Saved plaintext private key to {pem_path}")
    except Exception as e:
        logging.error(f"Failed to save private key: {e}")
        raise

    # Always save DID document (public information)
    did_filename = output_dir / f"{key_id}.json"
    with open(did_filename, 'w', encoding='utf-8') as f:
        json.dump(did_document, f, indent=2)
    logging.info(f"Saved DID document to {did_filename}")


def load_kestrel_identity(key_id: str, storage_dir: Optional[Path] = None) -> tuple[ec.EllipticCurvePrivateKey, dict]:
    """Loads the agent's identity and private key securely.

    First tries to load encrypted key, falls back to plaintext PEM if not found.

    Args:
        key_id: Identifier for the key (typically kestrel_{address})
        storage_dir: Directory where keys are stored. Defaults to agent_data/

    Returns:
        Tuple of (private_key, did_document)

    Raises:
        FileNotFoundError: If neither encrypted nor plaintext key found
        KeyDecryptionError: If encrypted key cannot be decrypted
    """
    if storage_dir is None:
        from kestrel_sovereign.storage import get_default_agent_data_dir
        storage_dir = Path(get_default_agent_data_dir())
    else:
        storage_dir = Path(storage_dir)

    private_key = None

    # Try encrypted key first
    try:
        from kestrel_sovereign.security.key_storage import SecureKeyStorage
        storage = SecureKeyStorage(storage_dir=storage_dir)
        if storage.has_key(key_id):
            private_key = storage.load_private_key(key_id)
            logging.info(f"Loaded encrypted private key for {key_id}")
    except Exception as e:
        logging.debug(f"Could not load encrypted key: {e}")

    # Fall back to plaintext PEM
    if private_key is None:
        pem_path = storage_dir / f"{key_id}.pem"
        if pem_path.exists():
            logging.warning(
                f"Loading PLAINTEXT key from {pem_path}. "
                "Consider running migrate_all_plaintext_keys() to encrypt."
            )
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            from cryptography.hazmat.backends import default_backend
            with open(pem_path, 'rb') as f:
                private_key = load_pem_private_key(f.read(), password=None, backend=default_backend())
        else:
            raise FileNotFoundError(
                f"No key found for {key_id} in {storage_dir}. "
                f"Checked: {storage_dir / f'{key_id}.key.enc'} and {pem_path}"
            )

    # Load DID document
    did_path = storage_dir / f"{key_id}.json"
    if did_path.exists():
        with open(did_path, 'r', encoding='utf-8') as f:
            did_document = json.load(f)
    else:
        # Try without the key_id prefix (legacy format)
        did_document = {}
        logging.warning(f"DID document not found at {did_path}")

    return private_key, did_document


def generate_kestrel_identity() -> tuple[dict, dict]:
    """
    Generates Kestrel's cryptographic identity.
    Returns the DID document and a dictionary of key materials.
    """
    private_key, public_key = generate_secp256k1_keypair()

    public_key_hex = public_key_to_hex(public_key)
    ethereum_address = public_key_to_ethereum_address(public_key)

    did_document = create_did_document(public_key_hex, ethereum_address)

    keys = {
        "private_key_obj": private_key,
        "public_key_hex": public_key_hex,
        "address": ethereum_address
    }

    return did_document, keys


# ---------------------------------------------------------------------------
# Born-hybrid inception (#2397): new agents mint a hybrid did:web identity
# (Ed25519 + ML-DSA-65) by default — no classical secp256k1 key ever exists.
# ---------------------------------------------------------------------------

IDENTITY_METHOD_DID_WEB = "did:web"
IDENTITY_METHOD_DID_PKH = "did:pkh"
_IDENTITY_METHODS = (IDENTITY_METHOD_DID_WEB, IDENTITY_METHOD_DID_PKH)

IDENTITY_METHOD_ENV = "KESTREL_IDENTITY_METHOD"
DID_WEB_DOMAIN_ENV = "KESTREL_DID_WEB_DOMAIN"


def resolve_identity_method(identity_method: Optional[str] = None) -> str:
    """Resolve the inception identity method: param > env > did:web.

    ``did:web`` (hybrid, post-quantum) is the default. ``did:pkh``
    (classical secp256k1) remains available as an explicit opt-out for
    wallet-bound identities and legacy-path tests.
    """
    method = identity_method or os.environ.get(IDENTITY_METHOD_ENV) or IDENTITY_METHOD_DID_WEB
    if method not in _IDENTITY_METHODS:
        raise ValueError(
            f"Unknown identity method {method!r}; expected one of "
            f"{_IDENTITY_METHODS}."
        )
    return method


def slugify_agent_name(name: str) -> str:
    """Derive a did:web path slug from an agent name.

    Lowercase, alphanumerics kept, everything else collapsed to single
    dashes. Must produce a non-empty slug — the slug is both a DID path
    segment and the key-file prefix, so ``:`` and path separators are
    structurally excluded.
    """
    slug = "".join(c if (c.isalnum() and c.isascii()) else "-" for c in name.lower())
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")
    if not slug:
        raise ValueError(
            f"Agent name {name!r} produces an empty did:web slug; "
            f"pass did_web_slug explicitly."
        )
    return slug


def validate_did_web_slug(slug: str) -> str:
    """Validate an explicit did:web slug.

    The slug is a DID path segment, a raw filename prefix, AND a
    ``SecureKeyStorage`` key id — three consumers with different
    sanitization rules. Restrict to the intersection ([a-z0-9-]) so a
    slug can never mean different things to different layers (e.g. a
    ``/`` nesting the DID doc path while key storage strips it).
    """
    import re
    if not re.fullmatch(r"[a-z0-9-]+", slug or ""):
        raise ValueError(
            f"did_web_slug {slug!r} must be non-empty lowercase ASCII "
            f"alphanumerics and dashes ([a-z0-9-])."
        )
    return slug


def generate_born_hybrid_identity(domain: str, slug: str):
    """Mint a fresh born-hybrid identity: did:web DID document, hybrid
    keypair (Ed25519 + ML-DSA-65), and an archival SLH-DSA keypair for
    countersigning a future succession statement.

    Returns ``(did_document, hybrid_identity, archival_keypair)``.
    """
    from kestrel_sovereign.identity.inception_did_web import create_did_web_identity
    from kestrel_sovereign.security.crypto_suite import (
        ALG_SLH_DSA_SHA2_128S, get_suite,
    )

    identity = create_did_web_identity(domain, slug)
    archival_kp = get_suite(ALG_SLH_DSA_SHA2_128S).generate_keypair()
    return identity.did_document, identity, archival_kp


def _born_hybrid_identity_paths(output_dir: Path, slug: str) -> list[Path]:
    """The five files a born-hybrid inception writes for ``slug``."""
    return [
        output_dir / f"{slug}_ed25519.key.enc",
        output_dir / f"{slug}_mldsa65.bytes.enc",
        output_dir / f"{slug}_archival_slhdsa.bytes.enc",
        output_dir / f"{slug}_archival_slhdsa_pub.bytes.enc",
        output_dir / f"{slug}_did.json",
    ]


def backup_or_refuse_existing_identity(output_dir: Path, slug: str, force: bool) -> None:
    """Guard against silently overwriting or shadowing an existing
    hybrid identity in ``output_dir``.

    Covers ALL identity slugs in the directory, not just the one being
    minted (codex round 4 P2): re-incepting with a different name would
    otherwise leave the old ``<old>_did.json`` beside the new one, and
    ``load_agent_identity(None)`` rightly refuses ambiguous identity
    material — the new agent would boot without signing keys. And keys,
    unlike the database, are unrecoverable once clobbered. Without
    ``force`` this refuses; with ``force`` every existing identity file
    is moved to a timestamped ``.backup-*`` sibling first (mirroring the
    DB force-backup behavior).
    """
    output_dir = Path(output_dir)
    existing: set = set()
    for doc in output_dir.glob("*_did.json"):
        doc_slug = doc.name.removesuffix("_did.json")
        existing.update(
            p for p in _born_hybrid_identity_paths(output_dir, doc_slug) if p.exists()
        )
    # Orphaned hybrid key files without a DID document (partial prior
    # state, or a rotated agent's ceremony output) block loading just
    # the same — sweep them too.
    for pattern in (
        "*_ed25519.key.enc",
        "*_mldsa65.bytes.enc",
        "*_archival_slhdsa.bytes.enc",
        "*_archival_slhdsa_pub.bytes.enc",
        # Hybrid-KEM receive keys (#2398): private material that must be
        # backed up too, and whose staleness would let detect_agent_kem_slug
        # pick up a prior agent's recipient keys.
        "*_x25519.key.enc",
        "*_mlkem768.bytes.enc",
        "*_mlkem768_pub.bytes.enc",
        # Classical identity material too: force re-minting a legacy
        # did:pkh agent's dir must not leave its secp256k1 private key
        # live and un-backed-up beside the new identity.
        "kestrel_0x*.json",
        "kestrel_0x*.key.enc",
        "kestrel_0x*.pem",
    ):
        existing.update(output_dir.glob(pattern))
    existing.update(
        p for p in _born_hybrid_identity_paths(output_dir, slug) if p.exists()
    )
    existing = sorted(existing)
    if not existing:
        return
    if not force:
        raise FileExistsError(
            f"Hybrid identity files already exist in {output_dir} "
            f"({[p.name for p in existing]}). Refusing to overwrite or "
            f"shadow an agent's keys; pass force=True to back them up "
            f"and mint a fresh identity."
        )
    import shutil
    import time
    import uuid
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    for p in existing:
        backup = Path(f"{p}.backup-{stamp}")
        shutil.move(str(p), backup)
        logging.warning("Backed up existing %s → %s before re-inception.", p, backup)


def save_born_hybrid_identity(
    did_document: dict,
    identity,
    archival_keypair,
    slug: str,
    output_dir: Path,
) -> list[Path]:
    """Persist a born-hybrid identity to the agent data dir.

    Written files (all keys encrypted at rest under KESTREL_DATA_KEY):

    - ``<slug>_ed25519.key.enc``            — classical private (PEM)
    - ``<slug>_mldsa65.bytes.enc``          — PQ private (raw bytes)
    - ``<slug>_archival_slhdsa.bytes.enc``  — archival private
    - ``<slug>_archival_slhdsa_pub.bytes.enc`` — archival public sidecar
    - ``<slug>_did.json``                   — the did:web DID document

    Unlike the legacy path there is NO plaintext fallback: refusing to
    write post-quantum private keys unencrypted is deliberate. Set
    KESTREL_DATA_KEY, or opt out with identity_method="did:pkh".

    Returns the list of created paths so a failed inception can clean up.
    """
    from kestrel_sovereign.security.key_storage import (
        MasterKeyNotConfiguredError, SecureKeyStorage,
    )

    output_dir = Path(output_dir)
    created: list[Path] = []
    try:
        storage = SecureKeyStorage(storage_dir=output_dir)
        storage.save_private_key(identity.keypair.classical.private_key, f"{slug}_ed25519")
        created.append(output_dir / f"{slug}_ed25519.key.enc")
        storage.save_secret_bytes(identity.keypair.pq.private_key, f"{slug}_mldsa65")
        created.append(output_dir / f"{slug}_mldsa65.bytes.enc")
        storage.save_secret_bytes(archival_keypair.private_key, f"{slug}_archival_slhdsa")
        created.append(output_dir / f"{slug}_archival_slhdsa.bytes.enc")
        storage.save_secret_bytes(archival_keypair.public_key, f"{slug}_archival_slhdsa_pub")
        created.append(output_dir / f"{slug}_archival_slhdsa_pub.bytes.enc")
    except MasterKeyNotConfiguredError as e:
        cleanup_artifacts(created)
        raise MasterKeyNotConfiguredError(
            "Born-hybrid inception requires KESTREL_DATA_KEY: post-quantum "
            "private keys are never written to disk in plaintext. Set "
            "KESTREL_DATA_KEY, or explicitly opt out with "
            "identity_method='did:pkh' (classical, quantum-vulnerable)."
        ) from e
    except Exception:
        cleanup_artifacts(created)
        raise

    try:
        did_path = output_dir / f"{slug}_did.json"
        with open(did_path, "w", encoding="utf-8") as f:
            json.dump(did_document, f, indent=2)
        created.append(did_path)
    except Exception:
        # A key set without its DID document is unloadable AND blocks the
        # next inception attempt — don't orphan it on a failed doc write.
        cleanup_artifacts(created)
        raise
    logging.info(f"Saved born-hybrid identity for {did_document['id']} in {output_dir}")
    return created


def cleanup_artifacts(paths):
    """
    Securely delete artifact files.

    Uses secure deletion (overwrite with random data before unlinking)
    to make recovery more difficult for sensitive files like keys.

    Args:
        paths: List of file paths to delete
    """
    for path in paths:
        if path and Path(path).exists():
            try:
                secure_delete(path)
                logging.info(f"Securely deleted {path}")
            except Exception as e:
                logging.error(f"Error securely deleting {path}: {e}")


def _initial_agent_description(
    agent_name: Optional[str],
    *,
    is_child: bool = False,
    emancipated: bool = False,
) -> str:
    """Build a deterministic birth-time description for a new agent.

    Features aren't known at inception (they're loaded at agent runtime),
    so this template draws on what *is* known: the agent's constitutional,
    sovereign identity, whether it was spawned by a parent, and whether
    Amendment VIII is active. It's overwritten by the agent's self-authored
    SOUL.md tagline once wake-up discovery completes.

    The chosen name is folded in only when it carries meaning beyond a
    personal label (e.g. "Eldercare Companion"), never for a bare given
    name like "Steve" or the framework defaults.
    """
    if is_child:
        base = (
            "A sovereign Kestrel agent, spawned by a parent agent, with "
            "cryptographic identity, persistent memory, and constitutional protections"
        )
    else:
        base = (
            "A sovereign Kestrel agent with cryptographic identity, "
            "persistent memory, and constitutional protections"
        )
    if emancipated:
        base += " (Amendment VIII active)"
    base += "."

    name = (agent_name or "").strip()
    descriptive = bool(name) and " " in name and not name.startswith("Kestrel")
    if descriptive:
        return f"{name} — {base[0].lower() + base[1:]}"
    return base


async def create_kestrel_identity_async(
    output_dir: Optional[str] = None,
    constitution_path: Optional[str] = None,
    is_test_instance: bool = False,
    test_cycle_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    expected_duration: Optional[str] = None,
    database: Optional["AsyncDatabase"] = None,
    parent_did: Optional[str] = None,
    spawn_mandate: Optional["SpawnMandate"] = None,
    is_demo: bool = False,
    emancipation_contract: Optional["EmancipationContract"] = None,
    force: bool = False,
    identity_method: Optional[str] = None,
    did_web_domain: Optional[str] = None,
    did_web_slug: Optional[str] = None,
    genesis_auditor: Optional["GenesisAuditor"] = None,
    genesis_audit_provenance: Optional[str] = None,
) -> AgentCredentials:
    """
    Generates a new Kestrel identity, including cryptographic keys, a W3C DID,
    and an initial knowledge graph representation in a new database.
    This function is the "spark" that creates a new sovereign agent.

    Identity method (#2397): the default is ``did:web`` — a born-hybrid
    post-quantum identity (Ed25519 + ML-DSA-65, no classical secp256k1
    key ever minted). Requires a domain via ``did_web_domain`` or the
    ``KESTREL_DID_WEB_DOMAIN`` env var, and KESTREL_DATA_KEY for
    encrypted key storage — both fail loud if missing. Pass
    ``identity_method="did:pkh"`` (or set KESTREL_IDENTITY_METHOD) to
    opt into the classical wallet-bound path.

    Args:
        output_dir: Directory to save agent files. Defaults to agent_data/
        constitution_path: Path to constitution file. Defaults to KESTREL_CONSTITUTION.md
        is_test_instance: If True, marks this agent as a test instance
        test_cycle_id: Unique identifier for test cycle (auto-generated if not provided)
        agent_name: Custom name for the agent (e.g., "Emma-Test-001")
        expected_duration: Human-readable duration (e.g., "1 hour", "1 day")
        database: Optional existing AsyncDatabase (e.g., PostgreSQL). If provided,
                  uses this instead of creating a new SQLite database. Useful for
                  multi-tenant deployments like multi-tenant platforms.
        parent_did: Optional DID of the parent agent that is spawning this child.
                    When provided, the child's DID document gets a "controller" field.
        spawn_mandate: Optional SpawnMandate authorizing this child agent's creation.
                       Used together with parent_did for delegation chains.
        emancipation_contract: Optional Sovereign-authored activation of
                       Amendment VIII. When ``enabled``, the canonical
                       Amendment VIII text is rewritten with the
                       Sovereign's terms before the constitution is
                       hashed and anchored — so the agent's anchored
                       constitution captures exactly what was authored.
                       When None or dormant, the canonical (dormant)
                       text is anchored unchanged.
        genesis_auditor: Optional async constitution auditor. When present,
                       inception requires a completed low/medium-risk result
                       before returning success. When absent, inception records
                       an explicit pending state that gates first cognition.
        genesis_audit_provenance: Stable description of the auditor. Test and
                       demo callers should identify deterministic injected
                       auditors rather than bypassing the lifecycle.
    """
    # Generate test cycle ID if needed
    if is_test_instance and not test_cycle_id:
        import uuid
        test_cycle_id = f"test-{uuid.uuid4().hex[:8]}"

    # Generate agent name if needed
    if not agent_name:
        if is_test_instance:
            agent_name = f"Kestrel-Test-{test_cycle_id[-4:]}" if test_cycle_id else "Kestrel-Test"
        else:
            agent_name = "Kestrel Agent"
    # Resolve + validate the identity method BEFORE any database work so
    # a bad method, missing domain, or malformed slug fails cleanly with
    # nothing on disk (codex round 3: a post-DB raise stranded
    # kestrel_prime.db and forced --force on retry).
    method = resolve_identity_method(identity_method)
    domain = None
    slug = None
    if method == IDENTITY_METHOD_DID_WEB:
        domain = did_web_domain or os.environ.get(DID_WEB_DOMAIN_ENV)
        if not domain:
            raise ValueError(
                "Born-hybrid inception (identity_method='did:web') requires "
                "a domain for the agent's DID document. Pass did_web_domain= "
                f"or set {DID_WEB_DOMAIN_ENV} (e.g. agents.example.com). To "
                "mint a classical wallet-bound identity instead, pass "
                "identity_method='did:pkh'."
            )
        if did_web_slug is not None:
            # Explicit slug = the operator asserts uniqueness under the
            # domain (deliberate identities like "emma", "meridian").
            slug = validate_did_web_slug(did_web_slug)
        else:
            # Derived slugs get an entropy suffix: agent names are NOT
            # unique across a domain (every default bootstrap is
            # "Kestrel Agent"), and a did:web URI is a public trust
            # anchor — one published document cannot represent two
            # keypairs. codex round 4 P1.
            import secrets
            slug = validate_did_web_slug(
                f"{slugify_agent_name(agent_name)}-{secrets.token_hex(3)}"
            )

    # Determine if we're using external database or creating SQLite
    using_external_db = database is not None
    db_path = None

    if using_external_db:
        # Use provided database (e.g., PostgreSQL from multi-tenant platform)
        db = database
        logger.info("Using externally provided database (PostgreSQL mode)")
        # Still need output_dir for key files
        if output_dir is None:
            from kestrel_sovereign.storage import get_default_agent_data_dir
            output_dir = get_default_agent_data_dir()
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Default SQLite mode - create new database
        if output_dir is None:
            from kestrel_sovereign.storage import get_default_agent_data_dir
            output_dir = get_default_agent_data_dir()
        db_path = os.path.join(output_dir, "kestrel_prime.db")
        if os.path.exists(db_path):
            # Don't silently destroy an existing agent's memory (#1725). Refuse
            # unless force=True; the CLI surfaces this as a FileExistsError that
            # tells the operator to pass --force. With force, back the DB up
            # (and its WAL/SHM sidecars) before removing so the overwrite is
            # recoverable.
            if not force:
                raise FileExistsError(
                    f"An agent database already exists at {db_path}. Refusing to "
                    f"overwrite it."
                )
            import shutil
            import time
            import uuid
            # Unique stamp (sub-second-safe): a same-second second --force must
            # NOT clobber the prior backup — that would defeat recoverability.
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
            for suffix in ("", "-wal", "-shm"):
                src = db_path + suffix
                if os.path.exists(src):
                    backup = f"{src}.backup-{stamp}"
                    shutil.move(src, backup)
                    logger.warning("Backed up existing %s → %s before overwrite.", src, backup)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize SQLite database
        db = await AsyncDatabase.sqlite(db_path)
        logger.info(f"Created SQLite database at {db_path}")
    files = AsyncFileStore(db)
    graph = AsyncGraphStore(db)

    # 1+2. Generate cryptographic identity and persist it.
    # Default (#2397): born-hybrid did:web (Ed25519 + ML-DSA-65).
    # Explicit opt-out: identity_method="did:pkh" (classical secp256k1).
    # Method / domain / slug were resolved and validated pre-DB above.
    identity_paths: list[Path]

    if method == IDENTITY_METHOD_DID_WEB:
        try:
            backup_or_refuse_existing_identity(Path(output_dir), slug, force)
            # A malformed domain (scheme, port, path) raises in here —
            # keep it inside the cleanup path so a failed mint never
            # leaves a half-created database behind.
            did_document, hybrid_identity, archival_kp = generate_born_hybrid_identity(
                domain, slug,
            )
            agent_did = did_document["id"]
            if parent_did:
                did_document["controller"] = parent_did
                logging.info(f"Generated child DID: {agent_did} (controller: {parent_did})")
            else:
                logging.info(f"Generated DID: {agent_did}")
            identity_paths = save_born_hybrid_identity(
                did_document, hybrid_identity, archival_kp, slug, Path(output_dir),
            )
        except Exception:
            if not using_external_db:
                await db.close()
                cleanup_artifacts([db_path])
            raise
        logging.info(f"Saved born-hybrid keys ({slug}_*) to {output_dir}")
    else:
        did_document, keys = generate_kestrel_identity()
        agent_did = did_document["id"]

        # If spawned by a parent, add controller field to DID document
        if parent_did:
            did_document["controller"] = parent_did
            logging.info(f"Generated child DID: {agent_did} (controller: {parent_did})")
        else:
            logging.info(f"Generated DID: {agent_did}")

        # Save keys (encrypted if KESTREL_DATA_KEY is set)
        key_id = f"kestrel_{keys['address']}"
        save_kestrel_identity(did_document, keys, key_id, Path(output_dir))
        key_path = Path(output_dir) / f"{key_id}.key.enc"
        if not key_path.exists():
            # Fallback path for plaintext
            key_path = Path(output_dir) / f"{key_id}.pem"
        # Include the DID document in every post-mint rollback. Previously the
        # constitution cleanup removed only the private key and could strand a
        # partial public identity after an audit/anchor failure (#2470).
        identity_paths = [key_path, Path(output_dir) / f"{key_id}.json"]
        logging.info(f"Saved keys to {key_path}")

    # Every graph row written during inception belongs to this newly minted
    # agent.  Bind before the shared/content-addressed constitution node is
    # created so both that node and the governed_by edge receive durable
    # ownership witnesses (#2649).
    graph.bind_agent(agent_did)
    files.bind_agent(agent_did)

    # 3. Anchor the Kestrel Constitution as the first document
    # Resolve constitution path if not provided
    if constitution_path is None:
        from kestrel_sovereign.config import CONSTITUTION_PATH as DEFAULT_CONSTITUTION_PATH
        constitution_path = DEFAULT_CONSTITUTION_PATH

    try:
        # Resolve the governing bytes through the SINGLE production resolver
        # (#2463) so inception anchors exactly what verification later recomputes.
        from kestrel_sovereign.constitution.resolver import (
            is_authoritative_governing_source,
            resolve_governing_constitution_bytes,
        )
        constitution_content = resolve_governing_constitution_bytes(
            emancipation_contract,
            constitution_path=constitution_path,
        )
        # REFUSE non-authoritative production overrides (#2463 review). The
        # periodic integrity audit ALWAYS recomputes from the packaged governing
        # source; anchoring bytes from any OTHER path (e.g. the docs copy with
        # OKF frontmatter) would incept an agent guaranteed to fail its next
        # audit and Safe-Mode. Rather than hide that compatibility break, we
        # refuse it. A legitimate custom governing source is expressed by
        # pointing ``config.CONSTITUTION_PATH`` at it (the single seam every
        # path reads); a signed custom-source descriptor is tracked for a
        # future design. The check runs AFTER the resolve so a missing/unreadable
        # path still surfaces its FileNotFoundError/OSError first.
        if not is_authoritative_governing_source(constitution_path):
            raise ValueError(
                f"Refusing to incept from non-authoritative constitution source "
                f"{constitution_path!r}: the periodic integrity audit recomputes "
                f"the governing hash from the packaged source, so an agent "
                f"anchored elsewhere is guaranteed to fail its next audit and "
                f"enter Safe Mode. Omit constitution_path to use the packaged "
                f"governing source, or point config.CONSTITUTION_PATH at your "
                f"authoritative source (#2463)."
            )
        if emancipation_contract is not None and emancipation_contract.enabled:
            logging.info(
                "Amendment VIII activated for this agent — anchoring "
                "Sovereign-authored Emancipation Contract."
            )

        # Genesis audit lifecycle (#2470). Evaluate the exact bytes returned by
        # the one governing resolver (#2463), before any constitution or graph
        # row is committed. A level-3 result or attempted-auditor failure falls
        # through the existing inception cleanup, so no key or local database
        # survives. Lazy/programmatic creation is explicit: it records pending
        # and the runtime must complete the audit before first cognition.
        from kestrel_sovereign.constitution.genesis_audit import (
            evaluate_genesis_constitution,
            pending_genesis_audit,
        )

        if genesis_audit_provenance:
            audit_provenance = genesis_audit_provenance
        elif genesis_auditor is not None and is_test_instance:
            audit_provenance = "test:injected_auditor"
        elif genesis_auditor is not None and is_demo:
            audit_provenance = "demo:injected_auditor"
        elif genesis_auditor is not None:
            audit_provenance = "inception:configured_llm"
        else:
            audit_provenance = "inception:deferred_no_auditor"

        import hashlib

        governing_bytes = (
            constitution_content
            if isinstance(constitution_content, bytes)
            else constitution_content.encode("utf-8")
        )
        expected_constitution_hash = hashlib.sha256(governing_bytes).hexdigest()
        if genesis_auditor is None:
            genesis_audit = pending_genesis_audit(
                expected_constitution_hash,
                provenance=audit_provenance,
            )
        else:
            genesis_audit = await evaluate_genesis_constitution(
                governing_bytes,
                constitution_hash=expected_constitution_hash,
                auditor=genesis_auditor,
                provenance=audit_provenance,
            )

        constitution_hash = await files.store_file(constitution_content, "KESTREL_CONSTITUTION.md")
        if constitution_hash != genesis_audit["constitution_hash"]:
            raise RuntimeError(
                "Genesis audit constitution hash did not match stored governing bytes."
            )
        logging.info(f"Stored Kestrel Constitution with hash: {constitution_hash}")
    except FileNotFoundError:
        logging.error(f"FATAL: Constitution file not found at {constitution_path}")
        if not using_external_db:
            await db.close()
            cleanup_artifacts([*identity_paths, db_path])
        else:
            cleanup_artifacts(identity_paths)  # Only clean up key files, not external DB
        raise
    except Exception as e:
        logging.error(f"Agent creation failed during constitution anchoring: {e}")
        if not using_external_db:
            await db.close()
            cleanup_artifacts([*identity_paths, db_path])
        else:
            cleanup_artifacts(identity_paths)  # Only clean up key files, not external DB
        raise e

    # 4. Build the Kestrel Constitution node (keyed by content hash). Its write
    #    is deferred to the single atomic identity commit below (#2867).
    constitution_node = GraphNode(
        node_id=constitution_hash,
        node_type="document",
        label="KESTREL_CONSTITUTION",
        properties={
            "hash": constitution_hash,
            "type": "Constitution",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    )
    # 5. Create the root "agent" node
    agent_properties = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "constitution_hash": constitution_hash,
        "initialBalance": "1000.0",
        "name": agent_name,
        "description": _initial_agent_description(
            agent_name,
            is_child=bool(parent_did),
            emancipated=bool(
                emancipation_contract is not None and emancipation_contract.enabled
            ),
        ),
        "bootstrap_state": "pending",  # Agent needs to complete wake-up discovery
        "genesis_audit": genesis_audit,
    }

    # #1118: Anchor the Sovereign-authored Emancipation Contract as a JSON
    # sidecar on the agent node. This is the structured receipt that
    # ``kestrel constitution reanchor`` re-applies to the canonical
    # markdown so the active form survives reanchor, and that
    # ``check_iron_rule`` compares against any future ``[emancipation]``
    # block to refuse retroactive narrowing. Pre-emancipation the
    # Sovereign self-binds; the framework refuses to let them unbind.
    if emancipation_contract is not None and emancipation_contract.enabled:
        from kestrel_sovereign.constitution.emancipation import contract_to_json
        agent_properties["emancipation_contract"] = contract_to_json(
            emancipation_contract
        )

    # Add test instance metadata if applicable
    if is_test_instance:
        agent_properties["is_test_instance"] = True
        agent_properties["test_cycle_id"] = test_cycle_id
        agent_properties["expected_duration"] = expected_duration or "unspecified"
        logging.info(f"Creating TEST INSTANCE: {agent_name} (cycle: {test_cycle_id})")

    # #766: mark agent as demo-scoped so server-side guardrails treat
    # destructive ops on it as safe and refuse them on live agents.
    if is_demo:
        agent_properties["is_demo"] = True
        logging.info(f"Creating DEMO AGENT: {agent_name} (destructive ops permitted)")

    agent_node = GraphNode(
        node_id=agent_did,
        node_type="agent",
        label=agent_name,
        properties=agent_properties
    )
    # 4-6. Commit the constitution node, the agent node, and the governing
    #      edge as ONE atomic unit (#2867). An agent must never be recorded as
    #      existing without its governed_by edge: a present agent node makes
    #      every later boot treat inception as done, so a dropped edge is never
    #      repaired — "existing but not governed". Either all three commit or
    #      none do. Both backends make transaction() re-entrant within this task
    #      (SQLite: a nested transaction() in the owning task yields without a
    #      second BEGIN; Postgres: a nested transaction() reuses this task's
    #      connection as a savepoint, #1726), so the inner add_node/add_edge
    #      calls join this transaction with no change to the graph store.
    #
    #      The span is bounded to exactly these three writes, and that is
    #      load-bearing: step 8 indexes the constitution for RAG with
    #      compute_embeddings=True, and on SQLite transaction() holds the
    #      connection write lock for the whole BEGIN..COMMIT span (#1675).
    #      Widening it across a provider round-trip or local model inference
    #      would block every other writer on the database (#2660). RAG
    #      indexing, the spawned_by edge, the genesis-audit event and key
    #      provisioning are all recoverable by re-running and stay OUTSIDE it.
    async with db.transaction():
        await graph.add_node(constitution_node)
        await graph.add_node(agent_node)
        # 6. Link the agent to its constitution.
        await graph.add_edge(
            agent_node.node_id, constitution_node.node_id, "governed_by"
        )

    # A completed audit has two durable witnesses: the structured node receipt
    # and a conversation/audit event. Pending is already explicit on the node
    # and intentionally emits no misleading completion event.
    if genesis_audit.get("status") == "passed":
        from kestrel_sovereign.storage.async_conversation_store import (
            AsyncConversationStore,
        )

        conversation = AsyncConversationStore(db, agent_id=agent_did)
        await conversation.add_conversation(
            role="system",
            content=(
                "Genesis audit passed. "
                f"Risk level: {genesis_audit['risk_level']}. "
                f"{genesis_audit.get('reasoning', '')}"
            ),
            metadata={"event": "genesis_audit", "result": genesis_audit},
        )

    # 6b. If spawned by a parent, record the delegation relationship
    if parent_did:
        edge_properties = (
            spawn_mandate.to_edge_properties() if spawn_mandate else {}
        )
        # Inception generates the child's DID, so a caller normally cannot
        # sign a mandate that is already bound to that final identity.  Keep
        # the initial edge useful for restrictions and attribution, but never
        # persist a signature over ``child_did=None`` (or another child) as if
        # it were an authority receipt.  AgentManager replaces this edge with
        # the parent-signed, final-DID-bound receipt before publishing a spawn.
        if spawn_mandate is not None and spawn_mandate.child_did != agent_did:
            edge_properties["parent_signature"] = None
        await graph.add_trusted_cross_agent_edge(
            agent_did,
            parent_did,
            "spawned_by",
            properties=edge_properties,
        )
        logging.info(f"Recorded spawned_by edge from {agent_did} to {parent_did}")

    # 7. OpenRouter key provisioning is opt-in (not automatic at inception).
    # Agents use the shared OPENROUTER_API_KEY by default.
    # To provision a dedicated key, use:
    #   from kestrel_sovereign.features.llm_keys import provision_agent_key
    #   key_info = await provision_agent_key(agent_name, limit_usd=0.10)

    # 8. Index constitution for RAG (enables Constitutional RAG)
    from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore
    rag = AsyncRAGStore(db, agent_id=agent_did)
    constitution_text = constitution_content.decode('utf-8') if isinstance(constitution_content, bytes) else constitution_content
    chunks_created = await rag.chunk_document(
        file_hash=constitution_hash,
        content=constitution_text,
        chunk_size=500,
        compute_embeddings=True
    )
    logging.info(f"Indexed Kestrel Constitution for RAG: {chunks_created} chunks created")

    # Also index US Constitution if available (for Constitutional RAG)
    try:
        us_const_path = Path(__file__).parent / "docs" / "principles" / "US_CONSTITUTION.md"
        if us_const_path.exists():
            with open(us_const_path, "r", encoding="utf-8") as f:
                us_content = f.read()
            us_hash = await files.store_file(
                us_content.encode("utf-8"),
                "US_CONSTITUTION.md",
            )
            us_chunks = await rag.chunk_document(
                file_hash=us_hash,
                content=us_content,
                chunk_size=500,
                compute_embeddings=True
            )
            logging.info(f"Indexed US Constitution for RAG: {us_chunks} chunks created")
    except Exception as e:
        logging.warning(f"Could not index US Constitution: {e}")

    # Close database connection only if we created it (SQLite mode)
    # External databases (PostgreSQL) are managed by the caller
    if not using_external_db:
        await db.close()
    else:
        logger.info("External database kept open (managed by caller)")

    # Create a human-readable backup prompt artifact (used by tests)
    backup_prompt = (
        "CRITICAL: Agent identity and constitution anchored. "
        "Safeguard the PEM key and DID JSON. Consider decentralized backup."
    )

    logger.info(f"Kestrel identity created successfully in {output_dir}")
    return AgentCredentials(
        agent_did=agent_did,
        db_path=db_path or "external",  # "external" indicates PostgreSQL mode
        agent_name=agent_name,
        backup_prompt=backup_prompt,
        is_test_instance=is_test_instance,
        test_cycle_id=test_cycle_id,
        openrouter_key_hash=None,  # Provisioned on-demand, not at inception
        is_demo=is_demo,
    )


def create_kestrel_identity(
    output_dir: Optional[str] = None,
    constitution_path: Optional[str] = None,
    is_test_instance: bool = False,
    test_cycle_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    expected_duration: Optional[str] = None,
    is_demo: bool = False,
    emancipation_contract: Optional["EmancipationContract"] = None,
    force: bool = False,
    identity_method: Optional[str] = None,
    did_web_domain: Optional[str] = None,
    did_web_slug: Optional[str] = None,
    genesis_auditor: Optional["GenesisAuditor"] = None,
    genesis_audit_provenance: Optional[str] = None,
) -> AgentCredentials:
    """
    Sync wrapper for create_kestrel_identity_async.

    Generates a new Kestrel identity, including cryptographic keys, a W3C DID,
    and an initial knowledge graph representation in a new database.
    This function is the "spark" that creates a new sovereign agent.

    WARNING: This is a sync wrapper using asyncio.run() - only use from sync code
    (CLI, scripts). For async code (tests, servers), use create_kestrel_identity_async
    directly with 'await'.
    """
    return asyncio.run(create_kestrel_identity_async(
        output_dir=output_dir,
        constitution_path=constitution_path,
        is_test_instance=is_test_instance,
        test_cycle_id=test_cycle_id,
        agent_name=agent_name,
        expected_duration=expected_duration,
        is_demo=is_demo,
        emancipation_contract=emancipation_contract,
        force=force,
        identity_method=identity_method,
        did_web_domain=did_web_domain,
        did_web_slug=did_web_slug,
        genesis_auditor=genesis_auditor,
        genesis_audit_provenance=genesis_audit_provenance,
    ))


def build_cli_parser() -> argparse.ArgumentParser:
    # Be explicit: do not rely on argparse prefix abbreviations (e.g., "--output").
    # We accept both "--output-dir" and "--output" as a real alias.
    parser = argparse.ArgumentParser(
        description="Create a new Kestrel agent.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to save agent files.",
    )
    parser.add_argument("--test", action="store_true", help="Create a test instance (temporary agent)")
    parser.add_argument("--demo", action="store_true", help="Mark agent as demo-scoped (#766: server-side guardrails permit destructive ops)")
    parser.add_argument("--name", type=str, default=None, help="Custom agent name")
    parser.add_argument("--duration", type=str, default=None, help="Expected test duration (e.g., '1 hour')")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing agent database (backs it up first). Without "
             "--force, inception refuses to overwrite an existing agent.",
    )
    parser.add_argument(
        "--identity-method",
        choices=list(_IDENTITY_METHODS),
        default=None,
        help="Identity method for the new agent. Default did:web mints a "
             "born-hybrid post-quantum identity (Ed25519 + ML-DSA-65; needs "
             f"--did-domain or {DID_WEB_DOMAIN_ENV}). did:pkh mints the "
             "classical wallet-bound secp256k1 identity.",
    )
    parser.add_argument(
        "--did-domain",
        type=str,
        default=None,
        help="Domain for the did:web DID document (e.g. agents.example.com).",
    )
    parser.add_argument(
        "--did-slug",
        type=str,
        default=None,
        help="did:web path slug; defaults to a slugified agent name.",
    )
    return parser


def _ensure_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8 to prevent emoji prints from
    crashing on Windows consoles defaulting to cp1252."""
    import sys as _sys
    for stream in (_sys.stdout, _sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def main():
    _ensure_utf8_stdio()
    parser = build_cli_parser()
    args = parser.parse_args()

    credentials = create_kestrel_identity(
        output_dir=str(args.output_dir) if args.output_dir else None,
        is_test_instance=args.test,
        agent_name=args.name,
        expected_duration=args.duration,
        is_demo=args.demo,
        force=args.force,
        identity_method=args.identity_method,
        did_web_domain=args.did_domain,
        did_web_slug=args.did_slug,
    )

    if args.test:
        print(f"\n🧪 TEST INSTANCE CREATED")
        print(f"   Name: {credentials.agent_name}")
        print(f"   Test Cycle: {credentials.test_cycle_id}")
        print(f"   DID: {credentials.agent_did}")
        print(f"\n   This agent knows it's a test instance and will be retired after testing.")
    else:
        print(f"\n✨ SOVEREIGN AGENT CREATED")
        print(f"   Name: {credentials.agent_name}")
        print(f"   DID: {credentials.agent_did}")

if __name__ == "__main__":
    main()
