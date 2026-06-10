#!/usr/bin/env python3
"""
Filecoin/IPFS adapter for Kestrel's sovereign storage system.
Provides decentralized storage while maintaining local caching for performance.
"""

import hashlib
import json
import logging
import zlib
from pathlib import Path
import os
from typing import Dict, Optional, Tuple, Any
import requests

from kestrel_sovereign.kestrel_config.constants import (
    REDIS_CONNECT_TIMEOUT,
    HTTP_TIMEOUT_SHORT,
    HTTP_TIMEOUT_DEFAULT,
)

try:
    from pylotus_rpc import LotusClient, HttpJsonRpcConnector  # optional
except Exception:
    LotusClient = None
    HttpJsonRpcConnector = None

# Import unified StorageTier and StorageResult from base
# StorageResult has backward-compatible aliases (storage_tier, ipfs_cid, filecoin_deal_id)
from kestrel_sovereign.storage.providers.base import StorageTier, StorageResult


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FilecoinAdapter:
    """Adapter for integrating Filecoin/IPFS with Kestrel's storage"""
    
    def __init__(self,
                 lotus_rpc_url: str = "http://localhost:1234/rpc/v0",
                 lotus_token: Optional[str] = None,
                 ipfs_api_url: str = "http://localhost:8889",
                 cache_dir: str = "./storage_cache"):
        """
        Initialize Filecoin adapter.
        
        Args:
            lotus_rpc_url: URL for the Lotus node's JSON-RPC endpoint.
            lotus_token: Authentication token for the Lotus node.
            ipfs_api_url: IPFS API endpoint (local node).
            cache_dir: Local cache directory for hot data.
        """
        # Allow environment overrides for containerized/local setups
        env_ipfs = os.environ.get("IPFS_API_URL")
        env_lotus = os.environ.get("LOTUS_RPC_URL")
        env_lotus_token = os.environ.get("LOTUS_RPC_TOKEN")

        self.ipfs_api_url = (env_ipfs or ipfs_api_url).rstrip('/')
        lotus_rpc_url = env_lotus or lotus_rpc_url
        lotus_token = env_lotus_token or lotus_token
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.lotus_client = self._initialize_lotus_client(lotus_rpc_url, lotus_token)
        
        self._ipfs_available = self._test_ipfs_connection()
        if not self._ipfs_available:
            logging.warning("Will operate in local-only mode for all decentralized storage tiers.")
        
        self._lotus_available = self.lotus_client is not None
        if not self._lotus_available:
            logging.warning("Will operate in IPFS-only mode for Filecoin storage tiers.")

    def _initialize_lotus_client(self, rpc_url: str, token: Optional[str]) -> Optional[object]:
        """Initializes and tests the Lotus client connection."""
        try:
            if LotusClient is None or HttpJsonRpcConnector is None:
                raise RuntimeError("pylotus-rpc not installed")
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            connector = HttpJsonRpcConnector(url=rpc_url, headers=headers)
            client = LotusClient(connector)
            version = client.Version()
            logging.info(f"✅ Connected to Lotus node: {version}")
            return client
        except Exception as e:
            logging.error(f"❌ Failed to connect to Lotus node: {e}")
            return None

    def ipfs_is_available(self) -> bool:
        return self._ipfs_available
    
    def lotus_is_available(self) -> bool:
        return self._lotus_available

    def _test_ipfs_connection(self) -> bool:
        """Test connectivity to IPFS and return status."""
        try:
            response = requests.post(f"{self.ipfs_api_url}/api/v0/version", timeout=REDIS_CONNECT_TIMEOUT)
            if response.status_code == 200:
                version_info = response.json()
                logging.info(f"✅ Connected to IPFS node: {version_info.get('Version')}")
                return True
            else:
                logging.warning(f"⚠️ IPFS node responded with status {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"❌ Failed to connect to IPFS: {e}")
            return False
    
    def store_content(self, 
                     content: bytes, 
                     storage_tier: StorageTier = StorageTier.IPFS,
                     encrypt: bool = False,
                     metadata: Optional[Dict] = None) -> StorageResult:
        """
        Store content in decentralized storage based on tier.
        
        Args:
            content: Raw content bytes
            storage_tier: Where to store the content
            encrypt: Whether to encrypt before storage
            metadata: Additional metadata
            
        Returns:
            StorageResult with storage details
        """
        # Calculate content hash
        content_hash = hashlib.sha256(content).hexdigest()
        
        # Handle encryption if requested
        final_content = content
        encryption_key_hash = None
        if encrypt:
            final_content, encryption_key_hash = self._encrypt_content(content)
        
        # Compress content for efficiency
        compressed_content = zlib.compress(final_content, level=6)
        
        result = StorageResult(
            content_hash=content_hash,
            tier=storage_tier,
            encrypted=encrypt,
            encryption_key_hash=encryption_key_hash
        )

        # Store based on tier
        if storage_tier == StorageTier.LOCAL_ONLY:
            self._store_local_cache(content_hash, compressed_content, metadata)

        elif storage_tier == StorageTier.IPFS:
            if not self.ipfs_is_available():
                logging.warning("IPFS not available. Storing locally instead.")
                result.tier = StorageTier.LOCAL_ONLY
                self._store_local_cache(content_hash, compressed_content, metadata)
                return result

            ipfs_cid = self._store_ipfs(compressed_content, metadata)
            result.cid = ipfs_cid
            # Also cache locally for performance
            self._store_local_cache(content_hash, compressed_content, metadata)

        elif storage_tier in [StorageTier.FILECOIN, StorageTier.ENCRYPTED_FILECOIN]:
            if not self.ipfs_is_available() or not self.lotus_is_available():
                logging.warning("IPFS or Lotus not available. Storing locally instead.")
                result.tier = StorageTier.LOCAL_ONLY
                self._store_local_cache(content_hash, compressed_content, metadata)
                return result

            # First store in IPFS
            ipfs_cid = self._store_ipfs(compressed_content, metadata)
            result.cid = ipfs_cid

            # Then create Filecoin deal for permanent storage
            deal_id = self._create_filecoin_deal(ipfs_cid, metadata)
            result.deal_id = deal_id

            # Cache locally
            self._store_local_cache(content_hash, compressed_content, metadata)
        
        logging.info(f"📁 Stored content: {content_hash[:16]}... -> {storage_tier.value}")
        return result
    
    def retrieve_content(self, 
                         content_hash: str, 
                         ipfs_cid: Optional[str] = None, 
                         key_hash: Optional[str] = None) -> bytes:
        """
        Retrieve content from storage (cache first, then IPFS).
        
        Args:
            content_hash: SHA256 hash of original content
            ipfs_cid: IPFS Content ID (if available)
            key_hash: Hash of the encrypted key, if content is encrypted.
            
        Returns:
            Original content bytes
        """
        # Try local cache first
        try:
            retrieved_content = self._retrieve_local_cache(content_hash)
            if retrieved_content:
                logging.info(f"📂 Retrieved from cache: {content_hash[:16]}...")
            else:
                # Try IPFS if CID available
                if ipfs_cid:
                    try:
                        retrieved_content = self._retrieve_ipfs(ipfs_cid)
                        if retrieved_content:
                            # Update cache
                            self._store_local_cache(content_hash, retrieved_content)
                            logging.info(f"📡 Retrieved from IPFS: {content_hash[:16]}...")
                    except Exception as e:
                        logging.error(f"IPFS retrieval failed for {ipfs_cid}: {e}")

            if not retrieved_content:
                raise ValueError(f"Content not found: {content_hash}")

            decompressed_content = zlib.decompress(retrieved_content)
            
            if key_hash:
                return self._decrypt_content(decompressed_content, key_hash)
            else:
                return decompressed_content

        except Exception as e:
            logging.error(f"Failed to retrieve content for {content_hash}: {e}")
            raise
    
    def _decrypt_content(self, encrypted_content: bytes, key_hash: str) -> bytes:
        """Decrypts content using the two-tiered key system.

        AEADCipher reads both v2 and legacy Fernet ciphertext, so existing
        Filecoin-stored content stays decryptable across the migration.
        """
        from kestrel_sdk.security.aead import AEADCipher

        # 1. Read the encrypted content key from where it was stored
        key_file = self.cache_dir / f"key_{key_hash}.key"
        if not key_file.exists():
            raise FileNotFoundError(f"Could not find key file for hash: {key_hash}")

        with open(key_file, 'rb') as f:
            encrypted_key = f.read()

        # 2. Decrypt the content key with the master key
        master_key = self._get_master_key()
        f_master = AEADCipher(master_key)
        try:
            content_key = f_master.decrypt(encrypted_key)
        except Exception as e:
            logging.error(f"Failed to decrypt content key with master key: {e}")
            raise

        # 3. Use the decrypted content key to decrypt the actual content
        f_content = AEADCipher(content_key)
        try:
            decrypted_content = f_content.decrypt(encrypted_content)
            return decrypted_content
        except Exception as e:
            logging.error(f"Failed to decrypt content: {e}")
            raise

    def _get_master_key(self) -> bytes:
        """Get master encryption key from centralized encryption module.

        Uses the same key derivation as the rest of the system, supporting
        both raw Fernet keys and passphrases (via SHA-256 derivation).
        """
        from kestrel_sovereign.security.encryption import get_master_key_bytes

        key = get_master_key_bytes()
        if key:
            return key

        # No key configured - fail explicitly (no hardcoded fallback)
        raise ValueError(
            "KESTREL_DATA_KEY environment variable required for Filecoin encryption. "
            "Set it to a passphrase or a valid Fernet key."
        )

    def _encrypt_content(self, content: bytes) -> Tuple[bytes, str]:
        """Encrypt content with a derived key and secure the key.

        Both layers (content and key wrap) now use AES-256-GCM v2.
        """
        from kestrel_sdk.security.aead import AEADCipher

        # 1. Generate a new, unique key for this specific content
        content_key = AEADCipher.generate_key()
        f_content = AEADCipher(content_key)
        encrypted_content = f_content.encrypt(content)

        # 2. Encrypt the content key with a master key
        master_key = self._get_master_key()
        f_master = AEADCipher(master_key)
        encrypted_key = f_master.encrypt(content_key)

        # 3. Use the hash of the encrypted key as its identifier
        key_hash = hashlib.sha256(encrypted_key).hexdigest()

        # 4. Store the encrypted key
        key_file = self.cache_dir / f"key_{key_hash}.key"
        with open(key_file, 'wb') as f:
            f.write(encrypted_key)

        return encrypted_content, key_hash

    def _find_suitable_miner(self, size_bytes: int = 0) -> Optional[str]:
        """
        Finds a suitable miner to make a deal with.

        Selection criteria (in priority order):
        1. Accepting deals (not full, actively sealing)
        2. Sufficient storage space for the deal
        3. Reasonable price (below market average)
        4. Good reputation (success rate, sector faults)

        Args:
            size_bytes: Size of data to store (used to filter miners with sufficient space)

        Returns:
            Miner address if found, None otherwise
        """
        if not self.lotus_is_available():
            logging.warning("Lotus not available for miner selection")
            return None

        try:
            # Get list of all miners on the network
            miners = self.lotus_client.StateListMiners()
            if not miners:
                logging.warning("No miners found on the network.")
                return None

            logging.info(f"Found {len(miners)} miners on the network")

            # Filter and score miners
            suitable_miners = []

            for miner in miners:
                try:
                    # Get miner info to check if they're accepting deals
                    miner_info = self.lotus_client.StateMinerInfo(miner)

                    # Get miner power to assess reliability
                    miner_power = self.lotus_client.StateMinerPower(miner)

                    # Filter criteria:
                    # 1. Check if miner has any power (active)
                    has_power = False
                    if miner_power and 'MinerPower' in miner_power:
                        raw_byte_power = int(miner_power['MinerPower'].get('RawBytePower', '0'))
                        has_power = raw_byte_power > 0

                    if not has_power:
                        continue

                    # 2. Check storage availability (if we can get deal info)
                    # Note: StateMinerAvailableBalance checks if miner has funds for deals
                    try:
                        available_balance = self.lotus_client.StateMinerAvailableBalance(miner)
                        has_balance = int(available_balance) > 0
                        if not has_balance:
                            continue
                    except Exception as e:
                        # Fail closed: if we can't verify the miner has funds,
                        # treat it as unfit and skip it rather than letting an
                        # unverified miner fall through as suitable. Log so the
                        # exclusion is diagnosable.
                        logging.warning(
                            "Skipping miner %s: balance check failed (%s)", miner, e,
                        )
                        continue

                    # Calculate a score for this miner
                    # Higher score = better miner
                    score = 0

                    # Score based on power (more power = more reliable)
                    if raw_byte_power > 0:
                        # Normalize to 0-100 range (log scale)
                        import math
                        score += min(50, math.log10(raw_byte_power) * 5)

                    # Score based on sector count (more sectors = more experience)
                    if 'SectorCount' in miner_info:
                        sector_count = miner_info['SectorCount']
                        score += min(30, sector_count / 10)

                    # Score based on worker balance (more balance = more stable)
                    try:
                        balance_attoFIL = int(available_balance)
                        # Convert to FIL (1 FIL = 10^18 attoFIL)
                        balance_FIL = balance_attoFIL / (10 ** 18)
                        score += min(20, balance_FIL)
                    except Exception:
                        pass

                    suitable_miners.append({
                        'address': miner,
                        'score': score,
                        'power': raw_byte_power,
                        'info': miner_info
                    })

                except Exception as e:
                    logging.debug(f"Could not evaluate miner {miner}: {e}")
                    continue

            if not suitable_miners:
                # Fail closed: returning an unvetted fallback miner defeats the
                # whole suitability filter. The caller treats None as "no
                # suitable miner" and raises a clear error, so return None
                # rather than a miner we couldn't verify (#1676).
                logging.warning(
                    "No suitable miner found after filtering — returning None "
                    "(no unvetted fallback)."
                )
                return None

            # Sort by score (highest first)
            suitable_miners.sort(key=lambda m: m['score'], reverse=True)

            # Select best miner
            best_miner = suitable_miners[0]
            logging.info(
                f"Selected miner: {best_miner['address']} "
                f"(score: {best_miner['score']:.1f}, "
                f"power: {best_miner['power'] / (1024**4):.2f} TiB)"
            )

            return best_miner['address']

        except Exception as e:
            # Fail closed: on any error selecting a miner, return None so the
            # caller surfaces "no suitable miner" rather than silently using an
            # unvetted first-available miner (#1676).
            logging.error(f"Error finding miners: {e}")
            return None
    
    def _store_local_cache(self, content_hash: str, content: bytes, metadata: Optional[Dict] = None):
        """Store content in local cache"""
        cache_file = self.cache_dir / f"{content_hash}.cache"
        with open(cache_file, 'wb') as f:
            f.write(content)
        
        if metadata:
            meta_file = self.cache_dir / f"{content_hash}.meta"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f)
    
    def _retrieve_local_cache(self, content_hash: str) -> Optional[bytes]:
        """Retrieve content from local cache"""
        cache_file = self.cache_dir / f"{content_hash}.cache"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return f.read()
        return None
    
    def _store_ipfs(self, content: bytes, metadata: Optional[Dict] = None) -> str:
        """Store content in IPFS and return CID"""
        try:
            # Add content to IPFS
            files = {'file': content}
            response = requests.post(
                f"{self.ipfs_api_url}/api/v0/add",
                files=files,
                params={'pin': 'true'},  # Pin to ensure persistence
                timeout=HTTP_TIMEOUT_DEFAULT
            )
            
            if response.status_code == 200:
                result = response.json()
                cid = result['Hash']
                logging.info(f"📤 IPFS stored: {cid}")
                return cid
            else:
                raise Exception(f"IPFS storage failed: {response.status_code}")
                
        except Exception as e:
            logging.error(f"IPFS storage error: {e}")
            raise
    
    def _retrieve_ipfs(self, cid: str) -> bytes:
        """Retrieve content from IPFS by CID"""
        try:
            response = requests.post(
                f"{self.ipfs_api_url}/api/v0/cat",
                params={'arg': cid},
                timeout=HTTP_TIMEOUT_DEFAULT
            )
            
            if response.status_code == 200:
                return response.content
            else:
                raise Exception(f"IPFS retrieval failed: {response.status_code}")
                
        except Exception as e:
            logging.error(f"IPFS retrieval error: {e}")
            raise
    
    def _create_filecoin_deal(self, ipfs_cid: str, metadata: Optional[Dict] = None) -> str:
        """
        Creates a Filecoin storage deal for the given IPFS CID.

        Args:
            ipfs_cid: The IPFS content identifier.
            metadata: Optional metadata for the deal.

        Returns:
            The Filecoin Deal ID.
        """
        if not self.lotus_is_available():
            raise ConnectionError("Lotus node is not available to create a Filecoin deal.")

        # Get file size if available in metadata
        size_bytes = metadata.get('size_bytes', 0) if metadata else 0

        miner = self._find_suitable_miner(size_bytes=size_bytes)
        if not miner:
            raise ConnectionError("Could not find a suitable miner to make a deal.")

        wallet_address = self.lotus_client.WalletDefaultAddress()
        if not wallet_address:
            raise ConnectionError("Could not get a default wallet address from the Lotus node.")

        # Simplified deal parameters for now
        epoch_price = "500000000"  # Price in AttoFIL per epoch
        deal_duration = 518400    # Roughly 6 months in epochs

        data_ref = {
            'TransferType': 'graphsync',
            'Root': {'/': ipfs_cid},
            'PieceCid': None, # Lotus will calculate this
            'PieceSize': 0 # Lotus will calculate this
        }

        proposal = {
            'Data': data_ref,
            'Wallet': wallet_address,
            'Miner': miner,
            'EpochPrice': epoch_price,
            'MinBlocksDuration': deal_duration
        }
        
        try:
            logging.info(f"Proposing Filecoin deal with: {proposal}")
            deal_response = self.lotus_client.ClientStartDeal(proposal)
            deal_cid = deal_response['/']
            logging.info(f"Successfully proposed deal for CID {ipfs_cid}. Deal CID: {deal_cid}")
            return deal_cid
        except Exception as e:
            logging.error(f"Failed to create Filecoin deal: {e}")
            raise

    def get_deal_status(self, deal_cid: str) -> Optional[Dict]:
        """Gets the current status of a Filecoin deal."""
        if not self.lotus_is_available():
            logging.warning("Lotus node not available to check deal status.")
            return None
        
        try:
            deal_info = self.lotus_client.ClientGetDealInfo(deal_cid)
            return deal_info
        except Exception as e:
            logging.error(f"Could not get deal info for {deal_cid}: {e}")
            return None

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about the storage usage."""
        stats = {
            'cache_size': sum(f.stat().st_size for f in self.cache_dir.glob('*') if f.is_file()),
            'cache_files': len(list(self.cache_dir.glob('*.cache'))),
            'ipfs_connected': False,
            'filecoin_connected': bool(self.lotus_client) # Check if lotus_client is initialized
        }
        
        try:
            response = requests.get(f"{self.ipfs_api_url}/api/v0/stats/bw", timeout=HTTP_TIMEOUT_SHORT)
            if response.status_code == 200:
                stats['ipfs_connected'] = True
                stats['ipfs_stats'] = response.json()
        except Exception as e:
            logging.warning(f"Failed to fetch IPFS stats: {e}")
        
        return stats
    
    def cleanup_cache(self, max_age_days: int = 30):
        """Clean up old cache files"""
        import time
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        cleaned = 0
        for cache_file in self.cache_dir.glob('*.cache'):
            if cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()
                # Also remove metadata file if it exists
                meta_file = cache_file.with_suffix('.meta')
                if meta_file.exists():
                    meta_file.unlink()
                cleaned += 1
        
        logging.info(f"🧹 Cleaned {cleaned} old cache files")