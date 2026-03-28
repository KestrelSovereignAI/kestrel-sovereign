#!/usr/bin/env python3
"""
Sovereign Storage Adapter V2 - The Encrypted Merkle Forest.

This adapter implements the "Convergent Sharding" protocol:
1. Shards data by time (Conversations) and content (Files).
2. Uses Convergent Encryption (Key = HMAC(Content)) for deduplication.
3. Manages a Root Manifest DAG on IPFS.
"""

import abc
import asyncio
import json
import logging
import hashlib
import hmac
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from kestrel_sovereign.filecoin_adapter import FilecoinAdapter, StorageTier, StorageResult
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.car_builder import CARBuilder, CARReader

# Constants
SHARD_SIZE_LIMIT = 5 * 1024 * 1024  # 5MB per shard
MANIFEST_VERSION = "3.0"

@dataclass
class ShardMetadata:
    """Metadata for a single encrypted shard"""
    shard_id: str
    type: str               # 'conversation', 'file', 'graph'
    time_range: str         # '2025-11'
    cid: str                # IPFS Content ID
    content_hash: str       # SHA256 of plaintext
    size_bytes: int
    encryption_algo: str = "AES-GCM-256"
    key_derivation: str = "HMAC-SHA256"

@dataclass
class AssetDescriptor:
    """Describes a binary asset to be included in an export.

    Provided by downstream ``AssetCollector`` implementations so
    kestrel-sovereign can encrypt/upload the asset alongside conversation
    shards.
    """
    asset_type: str          # 'avatar', 'lora_weights', 'selfie', 'personality'
    asset_key: str           # unique key within this agent (e.g. "avatar_main")
    content_hash: str        # SHA256 hex of the plaintext bytes
    size_bytes: int
    ipfs_cid: Optional[str] = None   # skip upload if already on IPFS
    data: Optional[bytes] = None     # raw bytes (None when ipfs_cid is set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    encrypted: bool = False          # True if data is pre-encrypted


@dataclass
class AssetMetadata:
    """Manifest entry for an exported asset (parallel to ShardMetadata)."""
    asset_type: str
    asset_key: str
    cid: str                 # IPFS CID (or local fallback)
    content_hash: str
    size_bytes: int
    encrypted: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class AssetCollector(abc.ABC):
    """Protocol for downstream apps to attach binary assets to an export.

    Implementations (e.g. Frinz companion avatars) return a list of
    ``AssetDescriptor`` objects that will be encrypted and bundled into
    the sovereignty export.
    """

    @abc.abstractmethod
    async def collect_assets(self, agent_did: str) -> List[AssetDescriptor]:
        """Return all assets that should be exported for *agent_did*."""
        ...


@dataclass
class RootManifest:
    """The Root DAG Node - The Agent's State"""
    version: str
    timestamp: str
    agent_did: str
    shards: List[ShardMetadata]
    assets: List[AssetMetadata] = field(default_factory=list)
    index_cid: Optional[str] = None
    keyring_cid: Optional[str] = None
    previous_root: Optional[str] = None # For git-like history

class ConvergentEncryptor:
    """
    Handles deterministic encryption for deduplication.
    Key = HMAC(Content, User_Secret)
    """
    def __init__(self, user_secret: str):
        self.secret = user_secret.encode('utf-8')

    def derive_key(self, content: bytes) -> bytes:
        """Derive a 32-byte key from content + secret"""
        h = hmac.new(self.secret, content, hashlib.sha256)
        return h.digest()

    def encrypt(self, content: bytes) -> Tuple[bytes, bytes]:
        """
        Encrypt content deterministically.
        Returns (ciphertext, key).
        """
        key = self.derive_key(content)
        aesgcm = AESGCM(key)
        # Deterministic IV is required for deduplication.
        # We use the first 12 bytes of the content hash as IV.
        # Security Note: This leaks equality (if IV+Ciphertext matches, plaintext matches).
        # This is acceptable and desired for deduplication.
        nonce = hashlib.sha256(content).digest()[:12]
        ciphertext = aesgcm.encrypt(nonce, content, None)
        return ciphertext, key

    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt content"""
        aesgcm = AESGCM(key)
        # We need to recover the nonce. In this scheme, we can't easily recover
        # the nonce from the ciphertext alone without storing it.
        # FIX: We should prepend the nonce to the ciphertext.
        nonce = ciphertext[:12]
        actual_ciphertext = ciphertext[12:]
        return aesgcm.decrypt(nonce, actual_ciphertext, None)

    def encrypt_with_nonce_prefix(self, content: bytes) -> Tuple[bytes, bytes]:
        """Encrypt and prepend nonce for storage"""
        key = self.derive_key(content)
        aesgcm = AESGCM(key)
        nonce = hashlib.sha256(content).digest()[:12]
        ciphertext = aesgcm.encrypt(nonce, content, None)
        return nonce + ciphertext, key


class SovereignStorageAdapter:
    """
    V2 Adapter for Sharded, Deduplicated, Encrypted Storage.
    """

    def __init__(self, db: AsyncDatabase, user_secret: str, filecoin_adapter: Optional[FilecoinAdapter] = None, agent_id: str = ""):
        self.db = db
        self.agent_id = agent_id
        self.encryptor = ConvergentEncryptor(user_secret)
        self.adapter = filecoin_adapter or FilecoinAdapter()
        self.logger = logging.getLogger(__name__)

    def _now_sql(self) -> str:
        """Get SQL expression for current timestamp based on backend type."""
        if self.db.backend_type == "postgres":
            return "NOW()"
        return "datetime('now')"

    async def _get_conversations(self) -> List[Dict]:
        """Get all conversations from DB for this agent"""
        rows = await self.db.fetchall(
            "SELECT role, content, metadata, id FROM conversation_history WHERE agent_id = ? ORDER BY id ASC",
            (self.agent_id,)
        )
        return [
            {
                "role": row[0],
                "content": row[1],
                "metadata": json.loads(row[2]) if row[2] else {},
                "id": row[3]
            }
            for row in rows
        ]

    async def _shard_conversations(self) -> Dict[str, List[Dict]]:
        """
        Groups conversations by Month (YYYY-MM).
        Returns { '2025-11': [msgs...], '2025-10': [msgs...] }
        """
        shards = {}
        conversations = await self._get_conversations()
        for msg in conversations:
            # Extract timestamp from metadata. Falls back to current time if missing.
            # Note: The conversation_history table has created_at, but we use metadata.timestamp
            # for sharding consistency since metadata is what gets exported/imported.
            ts_str = msg.get("metadata", {}).get("timestamp", datetime.now(UTC).isoformat())
            try:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                month_key = dt.strftime("%Y-%m")
            except ValueError:
                month_key = "unknown"

            if month_key not in shards:
                shards[month_key] = []
            shards[month_key].append(msg)
        return shards

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _upload_content(
        self, content: bytes, storage_tier: StorageTier, metadata: Optional[Dict] = None
    ) -> str:
        """Upload *content* and return its CID (or local fallback)."""
        result = await asyncio.to_thread(
            self.adapter.store_content,
            content=content,
            storage_tier=storage_tier,
            encrypt=False,
            metadata=metadata,
        )
        cid = result.ipfs_cid
        if not cid and result.storage_tier == StorageTier.LOCAL_ONLY:
            cid = f"local-{result.content_hash}"
        if not cid:
            raise RuntimeError("Failed to upload content")
        return cid

    async def _download_content(self, cid: str) -> bytes:
        """Download content by CID (handles local- fallback)."""
        return await asyncio.to_thread(
            self.adapter.retrieve_content,
            content_hash=cid.replace("local-", ""),
            ipfs_cid=cid if not cid.startswith("local-") else None,
        )

    def _encrypt_keyring(self, keyring: Dict[str, str]) -> bytes:
        """Encrypt the keyring dict and return nonce+ciphertext."""
        keyring_json = json.dumps(keyring).encode("utf-8")
        keyring_key = self.encryptor.derive_key(b"KESTREL_KEYRING_V2")
        aesgcm = AESGCM(keyring_key)
        nonce = os.urandom(12)
        return nonce + aesgcm.encrypt(nonce, keyring_json, None)

    def _decrypt_keyring(self, keyring_cipher: bytes) -> Dict[str, str]:
        """Decrypt a keyring blob and return the key map."""
        keyring_key = self.encryptor.derive_key(b"KESTREL_KEYRING_V2")
        aesgcm = AESGCM(keyring_key)
        nonce = keyring_cipher[:12]
        return json.loads(aesgcm.decrypt(nonce, keyring_cipher[12:], None).decode("utf-8"))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    async def export_agent(
        self,
        agent_did: str,
        storage_tier: StorageTier = StorageTier.IPFS,
        asset_collector: Optional[AssetCollector] = None,
    ) -> str:
        """
        Export agent state as a single CAR v1 archive.

        All encrypted shards, assets, and the keyring are packed into one
        CAR file via ``CARBuilder``.  A single ``store_content()`` call
        produces one CID that represents the entire export.

        1. Encrypt conversation shards → add as raw blocks
        2. Collect & encrypt assets → add as raw/external-ref blocks
        3. Encrypt keyring → add as raw block
        4. Build dag-cbor manifest → set as CAR root
        5. Upload single CAR blob → return one CID
        """
        self.logger.info(f"🌲 Starting V3 CAR Export for {agent_did}...")

        builder = CARBuilder()
        shard_metadata_list: List[ShardMetadata] = []
        asset_metadata_list: List[AssetMetadata] = []
        keyring: Dict[str, str] = {}

        # 1. Process Conversation Shards
        conv_shards = await self._shard_conversations()
        for month, msgs in conv_shards.items():
            shard_content = json.dumps(msgs).encode("utf-8")
            ciphertext, key = self.encryptor.encrypt_with_nonce_prefix(shard_content)

            block_cid = builder.add_raw_block(ciphertext)

            meta = ShardMetadata(
                shard_id=f"conv_{month}",
                type="conversation",
                time_range=month,
                cid=block_cid,
                content_hash=hashlib.sha256(shard_content).hexdigest(),
                size_bytes=len(ciphertext),
            )
            shard_metadata_list.append(meta)
            keyring[meta.shard_id] = key.hex()
            self.logger.info(f"   Shard {month}: {len(msgs)} msgs -> {block_cid[:20]}...")

        # 2. Process Assets
        if asset_collector is not None:
            descriptors = await asset_collector.collect_assets(agent_did)
            for desc in descriptors:
                if desc.ipfs_cid:
                    # Already on IPFS — store a lightweight link node
                    block_cid = builder.add_external_ref(desc.ipfs_cid, ref_type=desc.asset_type)
                    encrypted = False
                else:
                    if desc.data is None:
                        raise ValueError(
                            f"AssetDescriptor {desc.asset_key!r} has no data and no ipfs_cid"
                        )
                    if desc.encrypted:
                        asset_cipher = desc.data
                    else:
                        asset_cipher, asset_key = self.encryptor.encrypt_with_nonce_prefix(desc.data)
                        keyring[f"asset_{desc.asset_key}"] = asset_key.hex()
                    block_cid = builder.add_raw_block(asset_cipher)
                    encrypted = True

                asset_metadata_list.append(AssetMetadata(
                    asset_type=desc.asset_type,
                    asset_key=desc.asset_key,
                    cid=block_cid,
                    content_hash=desc.content_hash,
                    size_bytes=desc.size_bytes,
                    encrypted=encrypted,
                    metadata=desc.metadata,
                ))
                self.logger.info(f"   Asset {desc.asset_key}: {desc.asset_type} -> {block_cid[:20]}...")

        # 3. Encrypt & add keyring
        keyring_cipher = self._encrypt_keyring(keyring)
        keyring_cid = builder.add_raw_block(keyring_cipher)

        # 4. Build manifest as dag-cbor root
        manifest = RootManifest(
            version=MANIFEST_VERSION,
            timestamp=datetime.now(UTC).isoformat(),
            agent_did=agent_did,
            shards=shard_metadata_list,
            assets=asset_metadata_list,
            keyring_cid=keyring_cid,
        )
        manifest_cid = builder.add_dag_cbor_block(asdict(manifest))
        builder.set_root(manifest_cid)

        # 5. Upload single CAR blob
        car_bytes = builder.build()
        root_cid = await self._upload_content(car_bytes, storage_tier)

        self.logger.info(
            f"✅ V3 CAR Export Complete. {builder.block_count} blocks, "
            f"{len(car_bytes)} bytes -> {root_cid}"
        )
        return root_cid

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    @staticmethod
    def _manifest_from_dict(data: dict) -> RootManifest:
        """Reconstruct a ``RootManifest`` from a plain dict."""
        data["shards"] = [ShardMetadata(**s) for s in data["shards"]]
        data["assets"] = [AssetMetadata(**a) for a in data.get("assets", [])]
        return RootManifest(**data)

    async def import_agent(self, root_cid: str, target_db_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Import (restore) an agent from a single CAR archive CID.

        1. Download single CAR blob
        2. Parse with CARReader
        3. Extract & decrypt keyring
        4. Extract & decrypt shards
        5. Rebuild database
        6. Return stats including asset metadata for downstream apps

        Args:
            root_cid: The IPFS CID of the CAR archive
            target_db_path: Optional path for new database (defaults to self.db.db_path)

        Returns:
            Dict with import statistics and ``assets_restored`` list
        """
        self.logger.info(f"🌲 Starting V3 CAR Import from CID: {root_cid}")

        # 1. Download & parse CAR archive
        try:
            car_bytes = await self._download_content(root_cid)
            reader = CARReader(car_bytes)
            assert reader.verify(), "CAR archive block verification failed"
            self.logger.info(f"   CAR archive: {reader.block_count} blocks, verified")
        except Exception as e:
            raise RuntimeError(f"Failed to download/parse CAR archive: {e}")

        # 2. Extract manifest from root
        try:
            manifest_data = reader.get_dag_cbor_block(reader.root_cid)
            if manifest_data is None:
                raise ValueError("Root CID not found in CAR archive")
            manifest = self._manifest_from_dict(manifest_data)
            self.logger.info(f"   Manifest version {manifest.version} for agent {manifest.agent_did}")
        except Exception as e:
            raise RuntimeError(f"Failed to parse manifest: {e}")

        # 3. Decrypt keyring (from CAR block)
        try:
            keyring_cipher = reader.get_block(manifest.keyring_cid)
            if keyring_cipher is None:
                raise ValueError(f"Keyring block {manifest.keyring_cid} not in CAR")
            keyring = self._decrypt_keyring(keyring_cipher)
            self.logger.info(f"   Keyring decrypted: {len(keyring)} entries")
        except Exception as e:
            raise RuntimeError(f"Failed to decrypt Keyring: {e}")

        # 4. Decrypt shards (from CAR blocks)
        all_conversations: List[Dict] = []
        for shard_meta in manifest.shards:
            try:
                shard_cipher = reader.get_block(shard_meta.cid)
                if shard_cipher is None:
                    raise ValueError(f"Shard block {shard_meta.cid} not in CAR")

                shard_key_hex = keyring.get(shard_meta.shard_id)
                if not shard_key_hex:
                    raise ValueError(f"Missing key for shard {shard_meta.shard_id}")

                shard_key = bytes.fromhex(shard_key_hex)
                shard_json = self.encryptor.decrypt(shard_cipher, shard_key)
                shard_data = json.loads(shard_json.decode("utf-8"))

                all_conversations.extend(shard_data)
                self.logger.info(f"   Shard {shard_meta.shard_id}: {len(shard_data)} messages restored")
            except Exception as e:
                self.logger.error(f"Failed to restore shard {shard_meta.shard_id}: {e}")

        # 5. Rebuild database
        self.logger.info("   Rebuilding database...")
        await self.db.execute(
            "DELETE FROM conversation_history WHERE agent_id = ?",
            (self.agent_id,)
        )

        for msg in sorted(all_conversations, key=lambda m: m.get("id", 0)):
            metadata_json = json.dumps(msg.get("metadata", {}))
            await self.db.execute(
                f"INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, {self._now_sql()})",
                (self.agent_id, msg["role"], msg["content"], metadata_json),
            )

        stats: Dict[str, Any] = {
            "manifest_version": manifest.version,
            "agent_did": manifest.agent_did,
            "shards_restored": len(manifest.shards),
            "messages_restored": len(all_conversations),
            "assets_restored": [asdict(a) for a in manifest.assets],
            "timestamp": manifest.timestamp,
        }

        self.logger.info(
            f"✅ V3 CAR Import Complete. Restored {len(all_conversations)} messages, "
            f"{len(manifest.assets)} assets"
        )
        return stats
