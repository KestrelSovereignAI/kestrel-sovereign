"""
Filebase Storage Provider

Integrates with Filebase for S3-compatible IPFS storage. Filebase provides
a standard S3 API with automatic IPFS pinning and CID assignment — upload
via boto3 S3 calls, get IPFS CIDs back in response metadata.

Advantages over Lighthouse:
- Standard S3 API (boto3) — no custom SDK needed
- Automatic IPFS pinning on upload
- Content-addressed with IPFS CIDs
- S3-compatible retrieval with CID metadata

Pricing (as of March 2026):
- Free tier: 5 GB storage + 5 GB bandwidth
- Paid: Starting at $5.99/month for additional storage

Required environment variables:
    FILEBASE_API_KEY         - Filebase API key
    FILEBASE_API_KEY_SECRET  - Filebase API secret key
    FILEBASE_API_ENDPOINT    - S3 endpoint (default: https://s3.filebase.com)
    FILEBASE_BUCKET          - S3 bucket name (default: kestrel-sovereign)

Optional:
    KESTREL_CACHE_DIR - Local cache directory (default: ./storage_cache)

References:
- Filebase docs: https://docs.filebase.com/
- S3 API: Standard boto3 S3 client
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sdk.storage.providers.base import (
    StorageProvider,
    StorageResult,
    StorageTier,
)

logger = logging.getLogger(__name__)

# Pricing constants (USD)
FILEBASE_COST_PER_GB_MONTHLY = Decimal("0.05")  # Approximate cost per GB/month
FILEBASE_FREE_TIER_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB free tier


class FilebaseProvider(StorageProvider):
    """
    Filebase storage provider for Tier 3 (CLOUD_HOT) storage.

    Stores content on IPFS via Filebase's S3-compatible API using boto3.
    Content is automatically pinned to IPFS and retrievable via CID.

    Required environment variables:
        FILEBASE_API_KEY         - Filebase API key
        FILEBASE_API_KEY_SECRET  - Filebase API secret key
        FILEBASE_API_ENDPOINT    - S3 endpoint URL
        FILEBASE_BUCKET          - S3 bucket name

    Optional:
        KESTREL_CACHE_DIR - Local cache directory (default: ./storage_cache)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_key_secret: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        bucket_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Filebase provider.

        Args:
            api_key: Filebase API key (falls back to FILEBASE_API_KEY env var)
            api_key_secret: Filebase API secret key (falls back to FILEBASE_API_KEY_SECRET)
            endpoint_url: S3 endpoint URL (falls back to FILEBASE_API_ENDPOINT or default)
            bucket_name: S3 bucket name (falls back to FILEBASE_BUCKET or default)
            cache_dir: Local cache directory path (falls back to KESTREL_CACHE_DIR)
        """
        # Use provided values if not None, otherwise fall back to env vars
        self._api_key = api_key if api_key is not None else os.environ.get("FILEBASE_API_KEY", "")
        self._api_key_secret = api_key_secret if api_key_secret is not None else os.environ.get("FILEBASE_API_KEY_SECRET", "")
        self._endpoint_url = endpoint_url if endpoint_url is not None else os.environ.get(
            "FILEBASE_API_ENDPOINT", "https://s3.filebase.com"
        )
        self._bucket_name = bucket_name if bucket_name is not None else os.environ.get("FILEBASE_BUCKET", "kestrel-sovereign")

        # Local cache for performance
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(os.environ.get("KESTREL_CACHE_DIR", "./storage_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.cache_dir / "filebase_index.json"

        # Lazy-initialize boto3 S3 client
        self._s3_client = None
        self._available = False

        # Check for valid credentials (non-empty strings)
        has_credentials = bool(self._api_key and self._api_key_secret)

        if has_credentials:
            try:
                import boto3
                self._s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=self._api_key,
                    aws_secret_access_key=self._api_key_secret,
                    endpoint_url=self._endpoint_url,
                )
                # Test connection by checking if bucket exists, create if not
                self._ensure_bucket()
                self._available = True
                logger.info(
                    "FilebaseProvider initialized — bucket: %s  endpoint: %s",
                    self._bucket_name,
                    self._endpoint_url,
                )
            except ImportError:
                logger.warning("FilebaseProvider: boto3 not installed. Run: pip install boto3")
            except Exception as e:
                logger.warning("FilebaseProvider: failed to initialize S3 client: %s", e)
        else:
            missing = [
                v for v, val in [
                    ("FILEBASE_API_KEY", self._api_key),
                    ("FILEBASE_API_KEY_SECRET", self._api_key_secret),
                ] if not val
            ]
            logger.warning("FilebaseProvider: missing env vars: %s", ", ".join(missing))

    def _ensure_bucket(self) -> None:
        """Ensure the S3 bucket exists, create if not."""
        if not self._s3_client:
            return

        try:
            self._s3_client.head_bucket(Bucket=self._bucket_name)
            logger.debug("Bucket exists: %s", self._bucket_name)
        except Exception:
            # Bucket doesn't exist, try to create it
            try:
                self._s3_client.create_bucket(Bucket=self._bucket_name)
                logger.info("Created bucket: %s", self._bucket_name)
            except Exception as e:
                logger.warning("Failed to create bucket %s: %s", self._bucket_name, e)
                raise

    # ------------------------------------------------------------------
    # StorageProvider interface
    # ------------------------------------------------------------------

    @property
    def tier(self) -> StorageTier:
        return StorageTier.CLOUD_HOT

    @property
    def provider_name(self) -> str:
        return "filebase"

    def is_available(self) -> bool:
        return self._available and self._s3_client is not None

    async def store(
        self,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        encrypt: bool = True,
    ) -> StorageResult:
        """
        Store content in Filebase (S3-compatible IPFS storage).

        Args:
            content:  Raw content bytes
            metadata: Optional metadata dict (filename, content_type, tag, …)
            encrypt:  Whether to encrypt before storing (Fernet, same as other providers)

        Returns:
            StorageResult with CID and storage details
        """
        if not self.is_available():
            raise ConnectionError("FilebaseProvider is not available")

        metadata = metadata or {}
        content_hash = hashlib.sha256(content).hexdigest()
        filename = metadata.get("filename", f"{content_hash[:16]}.bin")

        # Determine file extension from filename
        ext = Path(filename).suffix or ".bin"

        # Generate S3 key: {agent_id}/{content_hash}.{ext}
        # For now, use content_hash as key (agent_id can be added later if needed)
        s3_key = f"{content_hash}{ext}"

        # Encrypt if requested
        final_content = content
        encryption_key_hash = None
        if encrypt:
            final_content, encryption_key_hash = await self._encrypt_content(content)

        # Upload to S3
        try:
            self._s3_client.put_object(
                Bucket=self._bucket_name,
                Key=s3_key,
                Body=final_content,
                ContentType=metadata.get("content_type", "application/octet-stream"),
                Metadata={
                    "content-hash": content_hash,
                    "original-filename": filename,
                }
            )
        except Exception as e:
            logger.error("Failed to upload to Filebase: %s", e)
            raise

        # Retrieve IPFS CID from object metadata
        # Filebase returns CID in the x-amz-meta-cid header after upload
        cid = None
        try:
            response = self._s3_client.head_object(
                Bucket=self._bucket_name,
                Key=s3_key,
            )
            # Filebase stores CID in response metadata
            cid = response.get("Metadata", {}).get("cid") or response.get("ResponseMetadata", {}).get("HTTPHeaders", {}).get("x-amz-meta-cid")

            # If CID not in metadata, use content_hash as fallback
            if not cid:
                logger.warning("CID not found in Filebase response, using content_hash")
                cid = content_hash
        except Exception as e:
            logger.warning("Failed to retrieve CID from Filebase: %s", e)
            cid = content_hash

        size = len(final_content)

        # Cache locally for fast re-retrieval
        await self._cache_content(s3_key, final_content)

        size_gb = Decimal(size) / Decimal(1024 * 1024 * 1024)
        storage_cost = size_gb * FILEBASE_COST_PER_GB_MONTHLY

        storage_result = StorageResult(
            content_hash=content_hash,
            cid=cid,
            tier=self.tier,
            provider=self.provider_name,
            encrypted=encrypt,
            encryption_key_hash=encryption_key_hash,
            size_bytes=size,
            content_type=metadata.get("content_type"),
            filename=filename,
            storage_cost_usd=storage_cost,
        )

        await self._update_index(storage_result)
        return storage_result

    async def retrieve(self, cid: str, encryption_key_hash: Optional[str] = None) -> bytes:
        """
        Retrieve content by CID or content hash.

        Args:
            cid:                  CID string or content hash
            encryption_key_hash:  Fernet key hash if content was encrypted

        Returns:
            Original (decrypted) content bytes
        """
        if not self.is_available():
            raise ConnectionError("FilebaseProvider is not available")

        # Try local cache first
        cached = await self._get_from_cache(cid)
        if cached:
            logger.debug("Retrieved from local cache: %s", cid[:20])
            content = cached
        else:
            logger.info("Fetching from Filebase: %s", cid[:20])

            # Try to find the object by iterating through index
            # In production, you'd want a better CID->key mapping
            s3_key = None
            index = await self._load_index()
            for item in index.values():
                if item.get("cid") == cid or item.get("content_hash") == cid:
                    # Reconstruct the S3 key
                    content_hash = item.get("content_hash")
                    filename = item.get("filename", "")
                    ext = Path(filename).suffix or ".bin"
                    s3_key = f"{content_hash}{ext}"
                    break

            if not s3_key:
                # Fallback: assume cid is the content_hash
                s3_key = cid if "." in cid else f"{cid}.bin"

            try:
                response = self._s3_client.get_object(
                    Bucket=self._bucket_name,
                    Key=s3_key,
                )
                content = response["Body"].read()
                await self._cache_content(s3_key, content)
            except Exception as e:
                logger.error("Failed to retrieve from Filebase: %s", e)
                raise

        if encryption_key_hash:
            content = await self._decrypt_content(content, encryption_key_hash)

        return content

    async def list_content(self, limit: int = 100, offset: int = 0) -> List[StorageResult]:
        """List stored content from the local index."""
        index = await self._load_index()
        items = list(index.values())[offset: offset + limit]
        return [StorageResult.from_dict(item) for item in items]

    async def delete(self, cid: str) -> bool:
        """
        Remove object from Filebase S3 bucket and local cache.

        Note: Content is content-addressed on IPFS and may still be
        accessible from other IPFS nodes/gateways.
        """
        if not self.is_available():
            return False

        # Find the S3 key from index
        s3_key = None
        index = await self._load_index()
        for item in index.values():
            if item.get("cid") == cid or item.get("content_hash") == cid:
                content_hash = item.get("content_hash")
                filename = item.get("filename", "")
                ext = Path(filename).suffix or ".bin"
                s3_key = f"{content_hash}{ext}"
                break

        if not s3_key:
            s3_key = cid if "." in cid else f"{cid}.bin"

        try:
            self._s3_client.delete_object(
                Bucket=self._bucket_name,
                Key=s3_key,
            )
        except Exception as e:
            logger.warning("Failed to delete from Filebase: %s", e)

        # Remove from index
        if cid in index:
            del index[cid]
            await self._save_index(index)

        # Remove from cache
        cache_file = self.cache_dir / f"{cid}.cache"
        if cache_file.exists():
            cache_file.unlink()

        logger.info("Removed from Filebase: %s", cid[:20])
        return True

    async def verify(self, cid: str) -> bool:
        """Verify content is accessible in Filebase."""
        if not self.is_available():
            return False

        # Find the S3 key from index
        s3_key = None
        index = await self._load_index()
        for item in index.values():
            if item.get("cid") == cid or item.get("content_hash") == cid:
                content_hash = item.get("content_hash")
                filename = item.get("filename", "")
                ext = Path(filename).suffix or ".bin"
                s3_key = f"{content_hash}{ext}"
                break

        if not s3_key:
            s3_key = cid if "." in cid else f"{cid}.bin"

        try:
            self._s3_client.head_object(
                Bucket=self._bucket_name,
                Key=s3_key,
            )
            return True
        except Exception as e:
            logger.warning("Verification failed for %s: %s", cid[:20], e)
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics including storage usage."""
        index = await self._load_index()
        total_size = sum(item.get("size_bytes", 0) for item in index.values())

        stats: Dict[str, Any] = {
            "tier": self.tier.value,
            "provider": self.provider_name,
            "available": self.is_available(),
            "total_items": len(index),
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024 * 1024 * 1024),
            "cost_per_gb_monthly": str(FILEBASE_COST_PER_GB_MONTHLY),
            "free_tier_bytes": FILEBASE_FREE_TIER_BYTES,
            "bucket": self._bucket_name,
            "endpoint": self._endpoint_url,
        }

        return stats

    async def estimate_cost(self, size_bytes: int) -> Decimal:
        """Estimate monthly storage cost. Returns $0 within the 5 GB free tier."""
        if size_bytes <= FILEBASE_FREE_TIER_BYTES:
            return Decimal("0")
        billable = size_bytes - FILEBASE_FREE_TIER_BYTES
        size_gb = Decimal(billable) / Decimal(1024 * 1024 * 1024)
        return size_gb * FILEBASE_COST_PER_GB_MONTHLY

    # ------------------------------------------------------------------
    # Encryption helpers (same pattern as other providers)
    # ------------------------------------------------------------------

    async def _encrypt_content(self, content: bytes) -> tuple:
        """Encrypt content using Fernet with a master-key-wrapped content key."""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            raise ImportError("cryptography package required for encryption")

        content_key = Fernet.generate_key()
        f = Fernet(content_key)
        encrypted = f.encrypt(content)

        master_key = self._get_master_key()
        f_master = Fernet(master_key)
        encrypted_key = f_master.encrypt(content_key)
        key_hash = hashlib.sha256(encrypted_key).hexdigest()

        key_file = self.cache_dir / f"key_{key_hash}.key"
        with open(key_file, "wb") as fh:
            fh.write(encrypted_key)

        return encrypted, key_hash

    async def _decrypt_content(self, encrypted: bytes, key_hash: str) -> bytes:
        """Decrypt content using stored Fernet key."""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            raise ImportError("cryptography package required for decryption")

        key_file = self.cache_dir / f"key_{key_hash}.key"
        if not key_file.exists():
            raise FileNotFoundError(f"Encryption key not found: {key_hash}")

        with open(key_file, "rb") as fh:
            encrypted_key = fh.read()

        master_key = self._get_master_key()
        f_master = Fernet(master_key)
        content_key = f_master.decrypt(encrypted_key)

        f_content = Fernet(content_key)
        return f_content.decrypt(encrypted)

    def _get_master_key(self) -> bytes:
        """Get master encryption key from centralized encryption module."""
        from kestrel_sdk.security.encryption import get_master_key_bytes

        key = get_master_key_bytes()
        if key:
            return key
        raise ValueError(
            "KESTREL_DATA_KEY environment variable is required for FilebaseProvider encryption."
        )

    # ------------------------------------------------------------------
    # Local cache / index helpers
    # ------------------------------------------------------------------

    async def _cache_content(self, key: str, content: bytes) -> None:
        cache_file = self.cache_dir / f"{key}.cache"
        with open(cache_file, "wb") as fh:
            fh.write(content)

    async def _get_from_cache(self, key: str) -> Optional[bytes]:
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            with open(cache_file, "rb") as fh:
                return fh.read()
        return None

    async def _load_index(self) -> Dict[str, Any]:
        import json
        if self._index_file.exists():
            with open(self._index_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    async def _save_index(self, index: Dict[str, Any]) -> None:
        import json
        with open(self._index_file, "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2)

    async def _update_index(self, result: StorageResult) -> None:
        index = await self._load_index()
        index[result.cid] = result.to_dict()
        await self._save_index(index)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_filebase_provider(
    api_key: Optional[str] = None,
    api_key_secret: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    bucket_name: Optional[str] = None,
) -> FilebaseProvider:
    """
    Create a FilebaseProvider from environment or explicit parameters.

    Args:
        api_key: Filebase API key (falls back to FILEBASE_API_KEY env var)
        api_key_secret: Filebase API secret key (falls back to FILEBASE_API_KEY_SECRET)
        endpoint_url: S3 endpoint URL (falls back to FILEBASE_API_ENDPOINT)
        bucket_name: S3 bucket name (falls back to FILEBASE_BUCKET)

    Returns:
        Configured FilebaseProvider
    """
    return FilebaseProvider(
        api_key=api_key,
        api_key_secret=api_key_secret,
        endpoint_url=endpoint_url,
        bucket_name=bucket_name,
    )
