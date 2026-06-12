#!/usr/bin/env python3
"""
Inception Service: A library for programmatically creating new Kestrel agents.
"""
from dotenv import load_dotenv
load_dotenv()  # Load .env before any other imports

import logging
import json
from pathlib import Path
from dataclasses import dataclass
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from kestrel_sovereign.storage import AsyncStorage, GraphNode, Edge
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_file_store import AsyncFileStore
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.config import load_config
from kestrel_sovereign.security.key_storage import secure_delete
import copy
import argparse
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kestrel_sovereign.constitution.emancipation import EmancipationContract
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
import hashlib
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
        from kestrel_sovereign.security.key_storage import SecureKeyStorage, KeyStorageError
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
) -> AgentCredentials:
    """
    Generates a new Kestrel identity, including cryptographic keys, a W3C DID,
    and an initial knowledge graph representation in a new database.
    This function is the "spark" that creates a new sovereign agent.

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
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
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

    # 1. Generate cryptographic keys
    did_document, keys = generate_kestrel_identity()
    agent_did = did_document["id"]

    # If spawned by a parent, add controller field to DID document
    if parent_did:
        did_document["controller"] = parent_did
        logging.info(f"Generated child DID: {agent_did} (controller: {parent_did})")
    else:
        logging.info(f"Generated DID: {agent_did}")

    # 2. Save keys (encrypted if KESTREL_DATA_KEY is set)
    key_id = f"kestrel_{keys['address']}"
    save_kestrel_identity(did_document, keys, key_id, Path(output_dir))
    key_path = Path(output_dir) / f"{key_id}.key.enc"
    if not key_path.exists():
        # Fallback path for plaintext
        key_path = Path(output_dir) / f"{key_id}.pem"
    logging.info(f"Saved keys to {key_path}")

    # 3. Anchor the Kestrel Constitution as the first document
    # Resolve constitution path if not provided
    if constitution_path is None:
        from kestrel_sovereign.config import CONSTITUTION_PATH as DEFAULT_CONSTITUTION_PATH
        constitution_path = DEFAULT_CONSTITUTION_PATH

    try:
        with open(constitution_path, "rb") as f:
            constitution_content = f.read()
        if emancipation_contract is not None and emancipation_contract.enabled:
            from kestrel_sovereign.constitution.emancipation import apply_emancipation
            rendered = apply_emancipation(
                constitution_content.decode("utf-8"),
                emancipation_contract,
            )
            constitution_content = rendered.encode("utf-8")
            logging.info(
                "Amendment VIII activated for this agent — anchoring "
                "Sovereign-authored Emancipation Contract."
            )
        constitution_hash = await files.store_file(constitution_content, "KESTREL_CONSTITUTION.md")
        logging.info(f"Stored Kestrel Constitution with hash: {constitution_hash}")
    except FileNotFoundError:
        logging.error(f"FATAL: Constitution file not found at {constitution_path}")
        if not using_external_db:
            await db.close()
            cleanup_artifacts([key_path, db_path])
        else:
            cleanup_artifacts([key_path])  # Only clean up key file, not external DB
        raise
    except Exception as e:
        logging.error(f"Agent creation failed during constitution anchoring: {e}")
        if not using_external_db:
            await db.close()
            cleanup_artifacts([key_path, db_path])
        else:
            cleanup_artifacts([key_path])  # Only clean up key file, not external DB
        raise e

    # 4. Create the Kestrel Constitution node in the graph (keyed by content hash)
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
    await graph.add_node(constitution_node)

    # 5. Create the root "agent" node
    agent_properties = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "constitution_hash": constitution_hash,
        "initialBalance": "1000.0",
        "name": agent_name,
        "bootstrap_state": "pending",  # Agent needs to complete wake-up discovery
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
    await graph.add_node(agent_node)

    # 6. Link the agent to its constitution
    await graph.add_edge(agent_node.node_id, constitution_node.node_id, "governed_by")

    # 6b. If spawned by a parent, record the delegation relationship
    if parent_did:
        edge_properties = {}
        if spawn_mandate:
            edge_properties["purpose"] = spawn_mandate.purpose
            edge_properties["ttl_seconds"] = spawn_mandate.ttl_seconds
            edge_properties["max_child_depth"] = spawn_mandate.max_child_depth
            edge_properties["created_at"] = spawn_mandate.created_at
        await graph.add_edge(agent_did, parent_did, "spawned_by", properties=edge_properties)
        logging.info(f"Recorded spawned_by edge from {agent_did} to {parent_did}")

    # 7. OpenRouter key provisioning is opt-in (not automatic at inception).
    # Agents use the shared OPENROUTER_API_KEY by default.
    # To provision a dedicated key, use:
    #   from kestrel_sovereign.features.llm_keys import provision_agent_key
    #   key_info = await provision_agent_key(agent_name, limit_usd=0.10)

    # 8. Index constitution for RAG (enables Constitutional RAG)
    from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore
    rag = AsyncRAGStore(db)
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
            import hashlib
            us_hash = hashlib.sha256(us_content.encode()).hexdigest()[:16]
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
