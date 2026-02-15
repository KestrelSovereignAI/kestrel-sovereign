#!/usr/bin/env python3
"""
Sovereign Storage Adapter V2 - The Encrypted Merkle Forest.

This adapter implements the "Convergent Sharding" protocol:
1. Shards data by time (Conversations) and content (Files).
2. Uses Convergent Encryption (Key = HMAC(Content)) for deduplication.
3. Manages a Root Manifest DAG on IPFS.
"""

import asyncio
import json
import logging
import hashlib
import hmac
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from kestrel_sovereign.filecoin_adapter import FilecoinAdapter, StorageTier, StorageResult
from kestrel_sovereign.storage.async_database import AsyncDatabase

# Constants
SHARD_SIZE_LIMIT = 5 * 1024 * 1024  # 5MB per shard
MANIFEST_VERSION = "2.0"

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
class RootManifest:
    """The Root DAG Node - The Agent's State"""
    version: str
    timestamp: str
    agent_did: str
    shards: List[ShardMetadata]
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

    async def export_agent(self, agent_did: str, storage_tier: StorageTier = StorageTier.IPFS) -> str:
        """
        The Main Export Loop.
        1. Shard Data
        2. Encrypt & Upload Shards
        3. Build Manifest
        4. Publish Root
        """
        self.logger.info(f"🌲 Starting V2 Export for {agent_did}...")

        shard_metadata_list = []
        keyring = {} # Map { shard_id: key_hex }

        # 1. Process Conversation Shards
        conv_shards = await self._shard_conversations()
        for month, msgs in conv_shards.items():
            shard_content = json.dumps(msgs).encode('utf-8')

            # Encrypt (Convergent)
            ciphertext, key = self.encryptor.encrypt_with_nonce_prefix(shard_content)

            # Upload
            result = await asyncio.to_thread(
                self.adapter.store_content,
                content=ciphertext,
                storage_tier=storage_tier,
                encrypt=False, # Already encrypted
                metadata={"type": "shard", "month": month}
            )

            # Handle Local Fallback (if IPFS is down)
            cid = result.ipfs_cid
            if not cid and result.storage_tier == StorageTier.LOCAL_ONLY:
                # Use content hash as pseudo-CID for local storage
                cid = f"local-{result.content_hash}"
                self.logger.warning(f"⚠️ IPFS unavailable. Using local CID: {cid}")

            if not cid:
                raise RuntimeError(f"Failed to upload shard {month}")

            # Record Metadata
            meta = ShardMetadata(
                shard_id=f"conv_{month}",
                type="conversation",
                time_range=month,
                cid=cid,
                content_hash=hashlib.sha256(shard_content).hexdigest(),
                size_bytes=len(ciphertext)
            )
            shard_metadata_list.append(meta)
            keyring[meta.shard_id] = key.hex()

            self.logger.info(f"   Shard {month}: {len(msgs)} msgs -> {cid}")

        # 2. Upload Keyring (Encrypted with User Secret)
        # The Keyring maps Shard IDs to their specific decryption keys.
        # We encrypt it using the User Secret so only the user can unlock the shards.
        # IMPORTANT: We use a FIXED known plaintext ("keyring") to derive the key,
        # so we can decrypt it later without needing the original keyring content.
        keyring_json = json.dumps(keyring).encode('utf-8')

        # Use a deterministic key based on a known constant
        keyring_encryption_key = self.encryptor.derive_key(b"KESTREL_KEYRING_V2")
        aesgcm = AESGCM(keyring_encryption_key)
        keyring_nonce = os.urandom(12)  # Random nonce is OK for keyring (not deduplicated)
        keyring_cipher = keyring_nonce + aesgcm.encrypt(keyring_nonce, keyring_json, None)

        keyring_result = await asyncio.to_thread(
            self.adapter.store_content, keyring_cipher, storage_tier
        )

        keyring_cid = keyring_result.ipfs_cid
        if not keyring_cid and keyring_result.storage_tier == StorageTier.LOCAL_ONLY:
             keyring_cid = f"local-{keyring_result.content_hash}"

        # 3. Build Root Manifest
        manifest = RootManifest(
            version=MANIFEST_VERSION,
            timestamp=datetime.now(UTC).isoformat(),
            agent_did=agent_did,
            shards=shard_metadata_list,
            keyring_cid=keyring_cid
        )

        manifest_json = json.dumps(asdict(manifest), indent=2).encode('utf-8')
        manifest_result = await asyncio.to_thread(
            self.adapter.store_content, manifest_json, storage_tier
        )

        root_cid = manifest_result.ipfs_cid
        if not root_cid and manifest_result.storage_tier == StorageTier.LOCAL_ONLY:
            root_cid = f"local-{manifest_result.content_hash}"

        self.logger.info(f"✅ V2 Export Complete. Root CID: {root_cid}")
        return root_cid

    async def import_agent(self, root_cid: str, target_db_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Import (restore) an agent from a Root CID.

        1. Download Root Manifest
        2. Download & Decrypt Keyring
        3. Download & Decrypt Shards
        4. Rebuild Database

        Args:
            root_cid: The IPFS CID of the Root Manifest
            target_db_path: Optional path for new database (defaults to self.db.db_path)

        Returns:
            Dict with import statistics
        """
        self.logger.info(f"🌲 Starting V2 Import from CID: {root_cid}")

        # 1. Download Root Manifest
        try:
            manifest_bytes = await asyncio.to_thread(
                self.adapter.retrieve_content,
                content_hash=root_cid.replace("local-", ""),  # Handle local fallback CIDs
                ipfs_cid=root_cid if not root_cid.startswith("local-") else None
            )
            manifest_data = json.loads(manifest_bytes.decode('utf-8'))
            # Manually reconstruct ShardMetadata objects from dicts
            shards_list = [ShardMetadata(**shard) for shard in manifest_data['shards']]
            manifest_data['shards'] = shards_list
            manifest = RootManifest(**manifest_data)
            self.logger.info(f"   Manifest version {manifest.version} for agent {manifest.agent_did}")
        except Exception as e:
            raise RuntimeError(f"Failed to download Root Manifest: {e}")

        # 2. Download & Decrypt Keyring
        try:
            keyring_cid = manifest.keyring_cid
            keyring_cipher = await asyncio.to_thread(
                self.adapter.retrieve_content,
                content_hash=keyring_cid.replace("local-", ""),
                ipfs_cid=keyring_cid if not keyring_cid.startswith("local-") else None
            )
            # Decrypt using the same deterministic key used during export
            keyring_key = self.encryptor.derive_key(b"KESTREL_KEYRING_V2")
            aesgcm = AESGCM(keyring_key)
            nonce = keyring_cipher[:12]
            encrypted_keyring = keyring_cipher[12:]
            keyring_json = aesgcm.decrypt(nonce, encrypted_keyring, None)
            keyring = json.loads(keyring_json.decode('utf-8'))
            self.logger.info(f"   Keyring decrypted: {len(keyring)} shards")
        except Exception as e:
            raise RuntimeError(f"Failed to decrypt Keyring: {e}")

        # 3. Download & Decrypt Shards
        all_conversations = []
        for shard_meta in manifest.shards:
            try:
                shard_cid = shard_meta.cid
                # For local fallback CIDs, extract the content hash
                if shard_cid.startswith("local-"):
                    # The local CID is "local-<content_hash>", but the actual cache key
                    # is the hash of the CIPHERTEXT, not the plaintext.
                    # We need to use the CID itself to retrieve.
                    content_hash_for_lookup = shard_cid.replace("local-", "")
                else:
                    content_hash_for_lookup = shard_meta.content_hash

                shard_cipher = await asyncio.to_thread(
                    self.adapter.retrieve_content,
                    content_hash=content_hash_for_lookup,
                    ipfs_cid=shard_cid if not shard_cid.startswith("local-") else None
                )

                # Get decryption key from keyring
                shard_key_hex = keyring.get(shard_meta.shard_id)
                if not shard_key_hex:
                    raise ValueError(f"Missing key for shard {shard_meta.shard_id}")

                shard_key = bytes.fromhex(shard_key_hex)
                shard_json = self.encryptor.decrypt(shard_cipher, shard_key)
                shard_data = json.loads(shard_json.decode('utf-8'))

                all_conversations.extend(shard_data)
                self.logger.info(f"   Shard {shard_meta.shard_id}: {len(shard_data)} messages restored")
            except Exception as e:
                self.logger.error(f"Failed to restore shard {shard_meta.shard_id}: {e}")
                # Continue with partial recovery

        # 4. Rebuild Database
        self.logger.info(f"   Rebuilding database...")

        # Clear existing conversations for this agent only
        await self.db.execute(
            "DELETE FROM conversation_history WHERE agent_id = ?",
            (self.agent_id,)
        )

        # Insert restored conversations
        for msg in sorted(all_conversations, key=lambda m: m.get('id', 0)):
            metadata_json = json.dumps(msg.get('metadata', {}))
            await self.db.execute(
                f"INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, {self._now_sql()})",
                (self.agent_id, msg['role'], msg['content'], metadata_json)
            )

        stats = {
            "manifest_version": manifest.version,
            "agent_did": manifest.agent_did,
            "shards_restored": len(manifest.shards),
            "messages_restored": len(all_conversations),
            "timestamp": manifest.timestamp
        }

        self.logger.info(f"✅ V2 Import Complete. Restored {len(all_conversations)} messages")
        return stats
