#!/usr/bin/env python3
"""
Filecoin/IPFS adapter for Kestrel's sovereign storage system.
Provides decentralized storage while maintaining local caching for performance.
"""

import hashlib
from enum import Enum
from io import BytesIO
import json
import logging
import zlib
from pathlib import Path
import os
from typing import Callable, Dict, Optional, Protocol, Tuple, Any
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

_RETRIEVAL_ENVELOPE_ALLOWANCE = 64 * 1024
_RETRIEVAL_CHUNK_BYTES = 64 * 1024
# The file name an IPFS upload carries when its caller names none.
_DEFAULT_UPLOAD_NAME = "content.bin"


class ContentRetrievalLimitError(ValueError):
    """A bounded storage retrieval exceeded its configured byte ceiling."""


def _encoded_retrieval_limit(max_output_bytes: Optional[int]) -> Optional[int]:
    """Bound compressed/encrypted bytes needed for a decoded output ceiling."""

    if max_output_bytes is None:
        return None
    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or max_output_bytes <= 0
    ):
        raise ValueError("max_output_bytes must be a positive integer")
    # AEAD envelopes can be larger than their plaintext.  Two times the final
    # ceiling plus a small fixed allowance keeps that intermediate bounded
    # without rejecting normal Fernet/AES envelopes near the caller's limit.
    return max_output_bytes * 2 + _RETRIEVAL_ENVELOPE_ALLOWANCE


def _decompress_retrieved_content(
    content: bytes,
    *,
    max_bytes: Optional[int],
) -> bytes:
    if max_bytes is None:
        return zlib.decompress(content)
    decompressor = zlib.decompressobj()
    decoded = decompressor.decompress(content, max_bytes + 1)
    if len(decoded) > max_bytes:
        raise ContentRetrievalLimitError(
            "retrieved content exceeds configured output limit"
        )
    if not decompressor.eof:
        raise zlib.error("compressed content is incomplete or invalid")
    return decoded


def _looks_like_sha256(value: str) -> bool:
    """True iff ``value`` is a 64-char lowercase-hex SHA-256 digest.

    Used to decide whether ``retrieve_content``'s integrity check applies — some
    callers pass an IPFS CID (base58 ``Qm…`` / base32 ``bafy…``) as the lookup
    key, which is NOT a sha256 and must not be hashed-compared (#1725)."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(c in "0123456789abcdef" for c in value.lower())


# HKDF ``info`` of the F187 per-content key. The envelope header names it, so
# both read from this one constant.
_CONTENT_KEY_INFO = b"kestrel-filecoin-content-key-v1"

# Self-describing encrypted envelope (#2873).
#
# The F187 per-content key is HKDF(KESTREL_DATA_KEY, salt=content hash), and a
# legacy encrypted payload carries no copy of that hash: it lived only in the
# exporter's local ``backup_artifact`` record, so an encrypted export could
# not be decrypted after the host or its database was lost, which is the
# disaster a backup exists for. An enveloped payload carries what decryption
# needs in an authenticated plaintext header:
#
#     ENVELOPE_MAGIC || header length (u32, big-endian) || header || KSAv2 token
#
# The header is canonical JSON naming the format and version, the cipher, the
# key derivation, ``key_hash`` (the content hash the key derives from, which
# is also the plaintext's SHA-256), ``key_id`` (a one-way identifier of the
# master key, so a payload sealed under another key is named as such instead
# of being reported as corrupt) and ``owner_id`` (a one-way identifier of the
# exporting agent's DID under the master key). It holds no key material, and
# the owner's DID is not readable from it without the master key. Everything
# before the token is the token's associated data, so changing any header byte
# fails decryption.
#
# ``owner_id`` exists because the master key is shared by every agent on a
# host, so it decrypts any of their envelopes. A restore that holds no
# receipt for the export has no other proof it is this agent's (#3225), and an
# export of another agent could not be restored usefully anyway: its
# conversation rows are encrypted at rest under that agent's DID.
ENVELOPE_MAGIC = b"KSENV\x00"
ENVELOPE_FORMAT = "kestrel.encrypted-envelope"
ENVELOPE_VERSION = 1
ENVELOPE_CIPHER = "aes-256-gcm"
ENVELOPE_KDF = "hkdf-sha256:" + _CONTENT_KEY_INFO.decode("ascii")
_ENVELOPE_LENGTH_BYTES = 4
_ENVELOPE_HEADER_LIMIT = 4096
_ENVELOPE_HEADER_FIELDS = frozenset(
    {"format", "version", "cipher", "kdf", "key_hash", "key_id", "owner_id"}
)
_ENVELOPE_KEY_ID_INFO = b"kestrel-encrypted-envelope-key-id-v1"
_ENVELOPE_KEY_ID_BYTES = 16
_ENVELOPE_OWNER_ID_INFO = b"kestrel-encrypted-envelope-owner-id-v1"
_ENVELOPE_OWNER_ID_BYTES = 16
# A legacy Fernet token is URL-safe base64 of version byte 0x80 followed by a
# big-endian timestamp; KSAv2 tokens carry their own prefix.
_FERNET_TOKEN_PREFIX = b"gAAAAA"


class EnvelopePolicy(Enum):
    """How ``FilecoinAdapter.retrieve_content`` treats what it retrieved.

    ``IGNORE`` never reads the payload as an envelope. ``OPEN`` opens an
    envelope from its own header and decrypts other ciphertext with
    ``key_hash``; legacy ciphertext that arrives without one is tried with the
    configured key under the name it was retrieved by, and plaintext is
    returned. ``ENCRYPTED_ONLY`` treats ciphertext as ``OPEN`` does and
    refuses plaintext.
    """

    IGNORE = "ignore"
    OPEN = "open"
    ENCRYPTED_ONLY = "encrypted_only"


class EncryptedContentError(ValueError):
    """Retrieved ciphertext could not be decrypted; none of it was returned."""


class DecryptionKeyUnavailableError(EncryptedContentError):
    """No master key is configured to decrypt the retrieved ciphertext with."""


class EncryptedEnvelopeError(EncryptedContentError):
    """An encrypted envelope could not be opened; nothing was decrypted."""


class EnvelopeFormatError(EncryptedEnvelopeError):
    """The envelope header is malformed or names a format this build cannot read."""


class EnvelopeKeyMismatchError(EncryptedEnvelopeError):
    """The envelope was sealed under another master key than the configured one."""

    def __init__(self, envelope_key_id: str, configured_key_id: str):
        self.envelope_key_id = envelope_key_id
        self.configured_key_id = configured_key_id
        super().__init__(
            f"the export was encrypted under key id {envelope_key_id}, but the "
            f"configured KESTREL_DATA_KEY has key id {configured_key_id}"
        )


class EnvelopeOwnerMismatchError(EncryptedEnvelopeError):
    """The envelope was sealed for another agent than the one opening it."""

    def __init__(self, envelope_owner_id: str, owner_id: str):
        self.envelope_owner_id = envelope_owner_id
        self.owner_id = owner_id
        super().__init__(
            f"the export was sealed for another agent (owner id "
            f"{envelope_owner_id}); this agent's owner id is {owner_id}"
        )


class EnvelopeDecryptionError(EncryptedEnvelopeError):
    """The envelope failed authentication or its plaintext failed its digest."""


class LegacyDecryptionUnavailableError(EncryptedContentError):
    """Legacy ciphertext with no key hash on record that the configured key
    did not decrypt.

    A legacy payload predates the envelope, so it does not carry the content
    hash its key derives from. ``attempted`` is True when the name it was
    retrieved by has the shape of that hash and the configured key failed
    authentication under it; False when the name is not a content hash, so
    no key could be derived to try.
    """

    def __init__(self, *, attempted: bool):
        self.attempted = attempted
        super().__init__(
            "the configured KESTREL_DATA_KEY does not decrypt it under the "
            "content hash it was requested by"
            if attempted
            else "its key derives from the export's content hash, which "
            "neither the payload nor the name it was requested by carries"
        )


class UnencryptedContentError(ValueError):
    """The payload is not encrypted, and the policy accepts only ciphertext."""

    def __init__(self) -> None:
        super().__init__("payload is not encrypted")


def is_encrypted_envelope(payload: bytes) -> bool:
    return payload.startswith(ENVELOPE_MAGIC)


def _is_legacy_ciphertext(payload: bytes) -> bool:
    from kestrel_sdk.security.aead import KSA_V2_PREFIX

    return payload.startswith(KSA_V2_PREFIX) or payload.startswith(_FERNET_TOKEN_PREFIX)


def envelope_key_id(master_key: bytes) -> str:
    """The one-way master-key identifier an envelope header records."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_ENVELOPE_KEY_ID_BYTES,
        salt=None,
        info=_ENVELOPE_KEY_ID_INFO,
    )
    return hkdf.derive(master_key).hex()


def envelope_owner_id(master_key: bytes, owner: str) -> str:
    """The one-way identifier of an envelope's owning agent DID.

    Keyed by the master key, so the header does not publish which agent an
    export belongs to.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    if not isinstance(owner, str) or not owner:
        raise ValueError("an encrypted envelope's owner must be an agent DID")
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_ENVELOPE_OWNER_ID_BYTES,
        salt=owner.encode("utf-8"),
        info=_ENVELOPE_OWNER_ID_INFO,
    )
    return hkdf.derive(master_key).hex()


def envelope_cache_name(content_hash: str, owner_id: str) -> str:
    """The host-cache name of an envelope sealed for ``owner_id``.

    Envelopes of one plaintext sealed for two agents share its content hash,
    and the host cache is one directory shared by every agent on the host.
    Keyed by the content hash alone, the second agent's envelope would
    overwrite the first's, whose restore would then be refused as another
    agent's export (#2873). ``owner_id`` is the ``owner_id`` the envelope's
    own header names.

    The owner id derives from the master key, so a reader under another key,
    or under none, cannot derive this name. ``store_content`` records it in
    ``StorageResult.cache_name``, and a restore reads the name its receipt
    records rather than deriving one.
    """
    return f"{content_hash}.{owner_id}"


def is_cache_name_of(name: Any, content_hash: str) -> bool:
    """Whether ``name`` is a host-cache name ``store_content`` writes content
    hashing to ``content_hash`` under: the hash itself, or the
    ``envelope_cache_name`` of an envelope of it.
    """
    if not isinstance(name, str) or not _is_lower_hex(content_hash, 64):
        return False
    stem, separator, owner_id = name.partition(".")
    if stem != content_hash:
        return False
    return not separator or _is_lower_hex(owner_id, _ENVELOPE_OWNER_ID_BYTES * 2)


def _envelope_header(*, key_hash: str, key_id: str, owner_id: str) -> bytes:
    header = {
        "format": ENVELOPE_FORMAT,
        "version": ENVELOPE_VERSION,
        "cipher": ENVELOPE_CIPHER,
        "kdf": ENVELOPE_KDF,
        "key_hash": key_hash,
        "key_id": key_id,
        "owner_id": owner_id,
    }
    return json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")


def _envelope_prefix(header: bytes) -> bytes:
    return ENVELOPE_MAGIC + len(header).to_bytes(_ENVELOPE_LENGTH_BYTES, "big") + header


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def _parse_envelope(payload: bytes) -> Tuple[Dict[str, Any], bytes, bytes]:
    """Split an envelope into its header, authenticated prefix and token."""
    length_start = len(ENVELOPE_MAGIC)
    header_start = length_start + _ENVELOPE_LENGTH_BYTES
    if not is_encrypted_envelope(payload) or len(payload) < header_start:
        raise EnvelopeFormatError("payload is not an encrypted envelope")
    header_length = int.from_bytes(payload[length_start:header_start], "big")
    header_end = header_start + header_length
    if not 0 < header_length <= _ENVELOPE_HEADER_LIMIT or len(payload) <= header_end:
        raise EnvelopeFormatError("encrypted envelope header length is invalid")
    raw_header = payload[header_start:header_end]
    try:
        header = json.loads(raw_header.decode("ascii"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        # Python 3.11's scanner raises RecursionError, not ValueError, for a
        # header of nested brackets well inside the length limit.
        raise EnvelopeFormatError("encrypted envelope header is not JSON") from exc
    if not isinstance(header, dict) or header.get("format") != ENVELOPE_FORMAT:
        raise EnvelopeFormatError("encrypted envelope header names no known format")
    version = header.get("version")
    if type(version) is not int or version != ENVELOPE_VERSION:
        raise EnvelopeFormatError(
            f"encrypted envelope version {version!r} is not supported "
            f"(this build reads version {ENVELOPE_VERSION})"
        )
    if set(header) != _ENVELOPE_HEADER_FIELDS:
        raise EnvelopeFormatError("encrypted envelope header fields are invalid")
    if header["cipher"] != ENVELOPE_CIPHER or header["kdf"] != ENVELOPE_KDF:
        raise EnvelopeFormatError(
            f"encrypted envelope cipher {header['cipher']!r} / key derivation "
            f"{header['kdf']!r} is not supported"
        )
    if not (
        _is_lower_hex(header["key_hash"], 64)
        and _is_lower_hex(header["key_id"], _ENVELOPE_KEY_ID_BYTES * 2)
        and _is_lower_hex(header["owner_id"], _ENVELOPE_OWNER_ID_BYTES * 2)
    ):
        raise EnvelopeFormatError("encrypted envelope key fields are invalid")
    canonical = _envelope_header(
        key_hash=header["key_hash"],
        key_id=header["key_id"],
        owner_id=header["owner_id"],
    )
    if canonical != raw_header:
        raise EnvelopeFormatError("encrypted envelope header is not canonical")
    return header, payload[:header_end], payload[header_end:]


class IpfsPinningService(Protocol):
    """A remote IPFS pinning service ``FilecoinAdapter.store_content`` can
    publish the IPFS tier to when the adapter's own IPFS node is unreachable.

    ``pin`` is called from the thread ``store_content`` runs in, with the
    payload exactly as the IPFS tier stores it (compressed, and sealed when
    encrypted), and returns its CID. ``provider_name`` is recorded as the
    result's provider.
    """

    provider_name: str

    def pin(
        self,
        payload: bytes,
        *,
        filename: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> str: ...


class _ProgressBytesIO(BytesIO):
    """BytesIO wrapper that reports file-byte progress as requests reads."""

    def __init__(
        self,
        content: bytes,
        on_progress: Optional[Callable[[int, int], None]],
        *,
        chunk_size: int = 64 * 1024,
    ):
        super().__init__(content)
        self._total = len(content)
        self._sent = 0
        self._on_progress = on_progress
        self._chunk_size = chunk_size
        self._report(0)

    def _report(self, sent: int) -> None:
        if self._on_progress is None:
            return
        try:
            self._on_progress(sent, self._total)
        except Exception:
            logging.debug("IPFS upload progress callback failed", exc_info=True)

    def read(self, size: int = -1) -> bytes:
        if size is None:
            size = -1
        if size is not None and size > self._chunk_size:
            size = self._chunk_size
        chunk = super().read(size)
        if chunk:
            self._sent += len(chunk)
            self._report(min(self._sent, self._total))
        return chunk


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
            logging.info("Will operate in local-only mode for all decentralized storage tiers.")

        self._lotus_available = self.lotus_client is not None
        if not self._lotus_available:
            logging.info("Will operate in IPFS-only mode for Filecoin storage tiers.")

    def _initialize_lotus_client(self, rpc_url: str, token: Optional[str]) -> Optional[object]:
        """Initializes and tests the Lotus client connection."""
        # pylotus-rpc is an optional dependency. Its absence is the normal case
        # for most deployments, so log at DEBUG rather than alarming at ERROR.
        if LotusClient is None or HttpJsonRpcConnector is None:
            logging.debug("Lotus/Filecoin support disabled: pylotus-rpc not installed.")
            return None
        try:
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
                     metadata: Optional[Dict] = None,
                     on_progress: Optional[Callable[[int, int], None]] = None,
                     *,
                     envelope_owner: Optional[str] = None,
                     upload_name: Optional[str] = None,
                     ipfs_pinning: Optional[IpfsPinningService] = None) -> StorageResult:
        """
        Store content in decentralized storage based on tier.

        Args:
            content: Raw content bytes
            storage_tier: Where to store the content
            encrypt: Whether to encrypt before storage
            metadata: Additional metadata
            on_progress: Optional callback(bytes_sent, total_bytes)
            envelope_owner: The exporting agent's DID. When given, the
                encrypted content is written as a self-describing envelope
                sealed for that agent, which decrypts from the payload alone
                plus KESTREL_DATA_KEY (#2873), and is cached under
                ``envelope_cache_name``. Requires ``encrypt``.
            upload_name: The file name an IPFS upload carries, recorded as
                the result's ``filename``. Defaults to ``content.bin``.
            ipfs_pinning: The IPFS tier's publisher when this adapter's IPFS
                node is unreachable, instead of storing locally only (#2874).
                Consulted for the IPFS tier alone. Requires ``upload_name``.

        Returns:
            StorageResult with storage details. Its ``cache_name`` is the
            name the content was cached under.
        """
        if envelope_owner is not None and not encrypt:
            raise ValueError("envelope_owner requires encrypt=True")
        if ipfs_pinning is not None and not upload_name:
            raise ValueError("ipfs_pinning requires the upload_name it pins under")

        # Calculate content hash
        content_hash = hashlib.sha256(content).hexdigest()

        # Handle encryption if requested
        final_content = content
        encryption_key_hash = None
        cache_name = content_hash
        if envelope_owner is not None:
            final_content, encryption_key_hash = self._seal_envelope(
                content, content_hash=content_hash, owner=envelope_owner
            )
            cache_name = envelope_cache_name(
                content_hash, envelope_owner_id(self._get_master_key(), envelope_owner)
            )
        elif encrypt:
            final_content, encryption_key_hash = self._encrypt_content(
                content, content_hash=content_hash
            )
        
        # Compress content for efficiency
        compressed_content = zlib.compress(final_content, level=6)
        self._report_progress(on_progress, 0, len(compressed_content))
        
        result = StorageResult(
            content_hash=content_hash,
            tier=storage_tier,
            cache_name=cache_name,
            encrypted=encrypt,
            encryption_key_hash=encryption_key_hash
        )

        # Store based on tier. Each local-cache write records the EFFECTIVE
        # tier/cid (result.tier/result.cid) so cleanup_cache knows whether a
        # remote replica exists (#1725).
        if storage_tier == StorageTier.LOCAL_ONLY:
            self._store_local_cache(cache_name, compressed_content, metadata,
                                    tier=result.tier, cid=result.cid)
            self._report_progress(on_progress, len(compressed_content), len(compressed_content))

        elif storage_tier == StorageTier.IPFS:
            if self.ipfs_is_available():
                ipfs_cid = self._store_ipfs(
                    compressed_content, metadata, on_progress=on_progress,
                    filename=upload_name or _DEFAULT_UPLOAD_NAME,
                )
            elif ipfs_pinning is not None:
                ipfs_cid = ipfs_pinning.pin(
                    compressed_content, filename=upload_name, on_progress=on_progress
                )
                result.provider = ipfs_pinning.provider_name
            else:
                logging.warning("IPFS not available. Storing locally instead.")
                result.tier = StorageTier.LOCAL_ONLY
                self._store_local_cache(cache_name, compressed_content, metadata,
                                        tier=result.tier, cid=result.cid)
                self._report_progress(on_progress, len(compressed_content), len(compressed_content))
                return result

            result.cid = ipfs_cid
            result.filename = upload_name
            # Also cache locally for performance
            self._store_local_cache(cache_name, compressed_content, metadata,
                                    tier=result.tier, cid=result.cid)

        elif storage_tier in [StorageTier.FILECOIN, StorageTier.ENCRYPTED_FILECOIN]:
            if not self.ipfs_is_available() or not self.lotus_is_available():
                logging.warning("IPFS or Lotus not available. Storing locally instead.")
                result.tier = StorageTier.LOCAL_ONLY
                self._store_local_cache(cache_name, compressed_content, metadata,
                                        tier=result.tier, cid=result.cid)
                self._report_progress(on_progress, len(compressed_content), len(compressed_content))
                return result

            # First store in IPFS
            ipfs_cid = self._store_ipfs(
                compressed_content, metadata, on_progress=on_progress,
                filename=upload_name or _DEFAULT_UPLOAD_NAME,
            )
            result.cid = ipfs_cid
            result.filename = upload_name

            # Then create Filecoin deal for permanent storage
            deal_id = self._create_filecoin_deal(ipfs_cid, metadata)
            result.deal_id = deal_id

            # Cache locally
            self._store_local_cache(cache_name, compressed_content, metadata,
                                    tier=result.tier, cid=result.cid)
        
        logging.info(f"📁 Stored content: {content_hash[:16]}... -> {storage_tier.value}")
        return result

    def _report_progress(
        self,
        on_progress: Optional[Callable[[int, int], None]],
        sent: int,
        total: int,
    ) -> None:
        if on_progress is None:
            return
        try:
            on_progress(sent, total)
        except Exception:
            logging.debug("Filecoin/IPFS upload progress callback failed", exc_info=True)
    
    def retrieve_content(self, 
                         content_hash: str, 
                         ipfs_cid: Optional[str] = None, 
                         key_hash: Optional[str] = None,
                         *,
                         max_output_bytes: Optional[int] = None,
                         allow_local_cache: bool = True,
                         envelope: EnvelopePolicy = EnvelopePolicy.IGNORE,
                         envelope_owner: Optional[str] = None,
                         envelope_key_hash: Optional[str] = None,
                         cache_name: Optional[str] = None) -> bytes:
        """
        Retrieve content from storage (cache first, then IPFS).

        ``allow_local_cache=False`` never consults the host's shared
        ``storage_cache`` and lets only ``ipfs_cid`` answer. The cache is one
        directory per host: a caller-supplied name that is really a co-hosted
        agent's content hash would otherwise be served from it — a LOCAL_ONLY
        export that was never published — or, failing that, tell the caller by
        its error whether the name exists (#3225). A caller with no ownership
        record for the name passes False.

        Under ``EnvelopePolicy.OPEN`` or ``ENCRYPTED_ONLY`` an encrypted
        envelope is decrypted from its own header with the configured master
        key, and only for ``envelope_owner``, the DID of the agent it was
        sealed for, which those policies require; the key hash the header
        names must equal ``envelope_key_hash`` (by default ``key_hash``) when
        one is given. The cache is read under ``cache_name`` when given, then
        under ``content_hash``. A payload without an envelope is decrypted
        with ``key_hash`` as before. Legacy ciphertext with no ``key_hash`` is
        tried with the configured key under ``content_hash`` (see
        ``_decrypt_legacy_under_name``), and raises
        ``LegacyDecryptionUnavailableError`` rather than being returned
        undecrypted. Plaintext is returned under ``OPEN`` and raises
        ``UnencryptedContentError`` under ``ENCRYPTED_ONLY``.

        Args:
            content_hash: SHA256 hash of original content
            ipfs_cid: IPFS Content ID (if available)
            key_hash: Hash of the encrypted key, if content is encrypted.
            max_output_bytes: Optional hard ceiling for the fully decoded
                result. Cache/network reads and decompression intermediates are
                bounded as well when supplied.
            envelope: How a self-describing encrypted envelope is treated.
            envelope_owner: The DID of the agent retrieving the content; an
                envelope sealed for another agent is refused.
            envelope_key_hash: The content hash the caller's record of the
                export names, which an envelope's header must name too.
                Defaults to ``key_hash``. A record that holds no key hash
                still names its export by content hash, so an envelope of
                another export is refused rather than restored under it.
            cache_name: The host-cache name the caller's record of the
                content holds (``StorageResult.cache_name``), read before
                ``content_hash``. An envelope is cached under a name
                that derives from the master key it was sealed under, so
                only the recorded name finds it when another key, or none,
                is configured; the envelope's header then names that key
                (#2873). The result must hash to the content hash the name
                begins with, from the cache or IPFS. Requires
                ``allow_local_cache``.

        Content fetched from IPFS is cached again under the name
        ``store_content`` gives it, not the name it was requested by, when
        the content establishes that name: an opened envelope's header, or
        the content hash it was checked against.

        Returns:
            Original content bytes
        """
        if envelope is not EnvelopePolicy.IGNORE and not envelope_owner:
            raise ValueError(
                f"envelope={envelope.value} requires the retrieving agent's "
                "DID as envelope_owner"
            )
        if envelope_key_hash is None:
            envelope_key_hash = key_hash
        if cache_name is not None:
            if not allow_local_cache:
                raise ValueError(
                    "cache_name names a host-cache entry, which "
                    "allow_local_cache=False forbids reading"
                )
            if not (
                isinstance(cache_name, str)
                and is_cache_name_of(cache_name, cache_name.partition(".")[0])
            ):
                raise ValueError("cache_name is not a name the host cache is written under")
        # Normalize a SHA-256 lookup key to lowercase so cache lookup AND the
        # integrity check are case-consistent on case-SENSITIVE filesystems too
        # (Linux CI/prod). Cache files are written with the lowercase hexdigest,
        # so an uppercase key would otherwise miss the file. CIDs are
        # case-sensitive and pass through unchanged (#1725 codex r3).
        if _looks_like_sha256(content_hash):
            content_hash = content_hash.lower()
        encoded_limit = _encoded_retrieval_limit(max_output_bytes)

        # Try local cache first (unless the caller may not use it)
        try:
            from_ipfs = False
            opened_header = None
            retrieved_content = None
            # The content hashes the result must hash to. A cache name begins
            # with the content hash of what is cached under it, whatever name
            # the content was requested by (an IPFS CID is not one), and
            # whether the cache or IPFS answers.
            expected_digests = {content_hash} if _looks_like_sha256(content_hash) else set()
            if cache_name is not None:
                expected_digests.add(cache_name.partition(".")[0])
            if allow_local_cache:
                # The recorded name first; then the name the content was
                # requested by, under which a restore by CID before #2873
                # cached what IPFS returned.
                lookup_names = [content_hash]
                if cache_name is not None and cache_name != content_hash:
                    lookup_names.insert(0, cache_name)
                for lookup_name in lookup_names:
                    retrieved_content = self._retrieve_local_cache(
                        lookup_name,
                        max_bytes=encoded_limit,
                    )
                    if retrieved_content:
                        break
            if retrieved_content:
                logging.info(f"📂 Retrieved from cache: {content_hash[:16]}...")
            else:
                # Try IPFS if CID available
                if ipfs_cid:
                    try:
                        retrieved_content = self._retrieve_ipfs(
                            ipfs_cid,
                            max_bytes=encoded_limit,
                        )
                        if retrieved_content:
                            from_ipfs = True
                            logging.info(f"📡 Retrieved from IPFS: {content_hash[:16]}...")
                    except ContentRetrievalLimitError:
                        raise
                    except Exception as e:
                        logging.error(f"IPFS retrieval failed for {ipfs_cid}: {e}")

            if not retrieved_content:
                raise ValueError(f"Content not found: {content_hash}")

            may_be_encrypted = bool(key_hash) or envelope is not EnvelopePolicy.IGNORE
            decompressed_content = _decompress_retrieved_content(
                retrieved_content,
                max_bytes=encoded_limit if may_be_encrypted else max_output_bytes,
            )

            if envelope is EnvelopePolicy.IGNORE:
                if key_hash:
                    result = self._decrypt_content(decompressed_content, key_hash)
                else:
                    result = decompressed_content
            elif is_encrypted_envelope(decompressed_content):
                opened_header, result = self._open_envelope(
                    decompressed_content,
                    owner=envelope_owner,
                    expected_key_hash=envelope_key_hash,
                )
            elif key_hash:
                result = self._decrypt_content(decompressed_content, key_hash)
            elif _is_legacy_ciphertext(decompressed_content):
                result = self._decrypt_legacy_under_name(
                    decompressed_content, content_hash
                )
            elif envelope is EnvelopePolicy.ENCRYPTED_ONLY:
                raise UnencryptedContentError()
            else:
                result = decompressed_content

            if max_output_bytes is not None and len(result) > max_output_bytes:
                raise ContentRetrievalLimitError(
                    "retrieved content exceeds configured output limit"
                )

            # Integrity check (#1725): when the lookup key is a genuine SHA-256
            # content hash, the retrieved bytes — especially from an untrusted
            # IPFS gateway — MUST hash back to it. We verify the fully-decoded
            # (decompressed + decrypted) result to detect cache poisoning /
            # gateway substitution, and only cache IPFS content AFTER it verifies.
            # NOTE: some callers retrieve by CID and pass ``content_hash=cid``
            # (a base58/base32 CID, NOT a sha256); those paths skip this check —
            # comparing a sha256 to a CID string would always (wrongly) fail —
            # unless the caller gave a recorded ``cache_name``, which names
            # its content hash.
            verified_digest = None
            if expected_digests:
                # content_hash was lowercased above, matching hexdigest().
                actual = hashlib.sha256(result).hexdigest()
                for expected in sorted(expected_digests):
                    if actual != expected:
                        raise ValueError(
                            f"Content integrity check failed for {expected[:16]}…: "
                            f"computed {actual[:16]}… (possible cache poisoning / gateway "
                            f"substitution)"
                        )
                verified_digest = actual

            if from_ipfs:
                # This content is durably on IPFS (we just fetched it by CID), so
                # the cache copy is safe to evict later. Content that
                # establishes its own name is cached under the name
                # ``store_content`` gives it, which a record of it holds, not
                # under the name it was requested by: an import requests an
                # export by its IPFS CID, and a copy under the CID is one its
                # receipt's name never finds (#2873). An opened envelope's
                # authenticated header names its content hash and the agent
                # it was sealed for; other content is named by the content
                # hash it was checked against. Content checked against none,
                # a CID lookup with no record, keeps the name it was
                # requested by, which the next lookup by that CID reads.
                if opened_header is not None:
                    recache_name = envelope_cache_name(
                        opened_header["key_hash"], opened_header["owner_id"]
                    )
                elif verified_digest is not None:
                    recache_name = verified_digest
                else:
                    recache_name = content_hash
                self._store_local_cache(
                    recache_name, retrieved_content,
                    tier=StorageTier.IPFS, cid=ipfs_cid,
                )

            return result

        except Exception as e:
            logging.error(f"Failed to retrieve content for {content_hash}: {e}")
            raise
    
    def _decrypt_content(self, encrypted_content: bytes, key_hash: str) -> bytes:
        """Decrypts content using the two-tiered key system.

        Two content-key provenance paths are supported:

        * **Portable (current, F187).** No key sidecar exists. The per-content
          key is *derived deterministically* from the master key + the content
          hash (``key_hash`` IS that content hash), so an encrypted CID
          restores on any host that shares ``KESTREL_DATA_KEY`` — no
          out-of-band sidecar file has to travel with the CID. This is what
          makes ``!identity import <cid> ...`` portable across substrates.
        * **Legacy sidecar.** Older content wrapped a *random* content key
          with the master key and wrote it to ``storage_cache/key_<hash>.key``.
          When that file is present we honour the old path so pre-existing
          Filecoin-stored content stays decryptable.

        AEADCipher reads both v2 and legacy Fernet ciphertext, so existing
        Filecoin-stored content stays decryptable across the migration.
        """
        from kestrel_sdk.security.aead import AEADCipher

        master_key = self._get_master_key()

        # Legacy path: a wrapped random content key was persisted as a local
        # sidecar. Only reachable on the ORIGINAL host that stored it — kept
        # for backward compatibility with content encrypted before F187.
        key_file = self.cache_dir / f"key_{key_hash}.key"
        if key_file.exists():
            with open(key_file, 'rb') as f:
                encrypted_key = f.read()
            try:
                content_key = AEADCipher(master_key).decrypt(encrypted_key)
            except Exception as e:
                logging.error(f"Failed to decrypt content key with master key: {e}")
                raise
        else:
            # Portable path: re-derive the deterministic per-content key from
            # the master key + content hash. No sidecar needed.
            content_key = self._derive_content_key(master_key, key_hash)

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

    def _derive_content_key(self, master_key: bytes, content_hash: str) -> bytes:
        """Derive a portable per-content AES-256 key from the master key.

        The derivation is deterministic in ``(master_key, content_hash)`` via
        HKDF-SHA256, so the same key is reproducible on any host that holds the
        same ``KESTREL_DATA_KEY`` — no key material has to be persisted to a
        sidecar or shipped out-of-band with the CID (F187 portability). The
        content hash is a stable, per-content salt so distinct packages get
        distinct content keys even under one master key.
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,  # AES-256
            salt=content_hash.encode("utf-8"),
            info=_CONTENT_KEY_INFO,
        )
        return hkdf.derive(master_key)

    def _seal_envelope(self, content: bytes, content_hash: str,
                       owner: str) -> Tuple[bytes, str]:
        """Encrypt content into a self-describing envelope (#2873).

        The key is the same F187 per-content key ``_encrypt_content`` uses;
        the envelope differs only in carrying ``key_hash``, the master-key
        identifier and the owning agent's identifier in its authenticated
        header, so it decrypts from the payload and KESTREL_DATA_KEY alone,
        for the agent ``owner`` names.
        """
        from kestrel_sdk.security.aead import AEADCipher

        master_key = self._get_master_key()
        prefix = _envelope_prefix(
            _envelope_header(
                key_hash=content_hash,
                key_id=envelope_key_id(master_key),
                owner_id=envelope_owner_id(master_key, owner),
            )
        )
        content_key = self._derive_content_key(master_key, content_hash)
        token = AEADCipher(content_key).encrypt(content, aad=prefix)
        return prefix + token, content_hash

    def open_envelope(self, payload: bytes, *, owner: str,
                      expected_key_hash: Optional[str] = None) -> bytes:
        """Decrypt a self-describing envelope with the configured master key.

        ``owner`` is the DID of the agent opening the envelope. Every
        co-hosted agent shares the master key, so it would decrypt another
        agent's envelope just as readily; one sealed for another agent is
        refused before decryption.

        ``expected_key_hash`` is the key hash a caller's own record holds for
        this payload, if it has one. An envelope naming another one is a
        different export, so it is refused before decryption too.

        Raises an ``EncryptedEnvelopeError`` subclass, and returns nothing,
        when the header is malformed or names another key hash than
        ``expected_key_hash``, the envelope names another master key or was
        sealed for another agent, or the token fails authentication or its
        plaintext does not hash to the header's ``key_hash``. Raises
        ``DecryptionKeyUnavailableError`` when no master key is configured.
        """
        return self._open_envelope(
            payload, owner=owner, expected_key_hash=expected_key_hash
        )[1]

    def _open_envelope(self, payload: bytes, *, owner: str,
                       expected_key_hash: Optional[str] = None
                       ) -> Tuple[Dict[str, Any], bytes]:
        """``open_envelope``, also returning the header it authenticated."""
        from kestrel_sdk.security.aead import AEADCipher
        from kestrel_sdk.security.exceptions import DecryptionError

        header, prefix, token = _parse_envelope(payload)
        if expected_key_hash is not None and header["key_hash"] != expected_key_hash:
            raise EnvelopeDecryptionError(
                f"the export's envelope names content hash {header['key_hash']}, "
                f"but the record of this export holds {expected_key_hash}"
            )
        master_key = self._require_master_key()
        configured_key_id = envelope_key_id(master_key)
        if header["key_id"] != configured_key_id:
            raise EnvelopeKeyMismatchError(header["key_id"], configured_key_id)
        owner_id = envelope_owner_id(master_key, owner)
        if header["owner_id"] != owner_id:
            raise EnvelopeOwnerMismatchError(header["owner_id"], owner_id)
        content_key = self._derive_content_key(master_key, header["key_hash"])
        try:
            content = AEADCipher(content_key).decrypt(token, aad=prefix)
        except DecryptionError as exc:
            raise EnvelopeDecryptionError(
                "the encrypted export failed authentication: it is corrupted "
                "or was tampered with"
            ) from exc
        if hashlib.sha256(content).hexdigest() != header["key_hash"]:
            raise EnvelopeDecryptionError(
                "the decrypted export does not match the content hash its "
                "envelope names"
            )
        return header, content

    def _decrypt_legacy_under_name(self, payload: bytes, name: str) -> bytes:
        """Decrypt legacy ciphertext that no record names a key hash for.

        A legacy payload predates the envelope (#2873). Its F187 key is
        HKDF(master key, salt=content hash), and the payload does not carry
        that hash. With no record of it, the one candidate is the name the
        payload was retrieved by, which is that hash when the export is named
        by it: ``export_sovereignty`` reports the content hash as the CID of
        an export no IPFS node returned a CID for. An IPFS CID is never a
        content hash, so a payload retrieved by one has no key to try.

        No key sidecar is read. A pre-F187 export's random key lives only in
        a ``key_<hash>.key`` sidecar in the host cache, which co-hosted agents
        share, and a caller without a record of the export may not read that
        cache (#3225).

        Raises ``LegacyDecryptionUnavailableError`` when the name is not a
        content hash or the configured key fails authentication under it,
        and ``DecryptionKeyUnavailableError`` when no master key is
        configured. The caller still checks the plaintext against the name.
        """
        from kestrel_sdk.security.aead import AEADCipher
        from kestrel_sdk.security.exceptions import DecryptionError

        if not _is_lower_hex(name, 64):
            raise LegacyDecryptionUnavailableError(attempted=False)
        master_key = self._require_master_key()
        content_key = self._derive_content_key(master_key, name)
        try:
            return AEADCipher(content_key).decrypt(payload)
        except DecryptionError as exc:
            raise LegacyDecryptionUnavailableError(attempted=True) from exc

    def _require_master_key(self) -> bytes:
        """The master key, or ``DecryptionKeyUnavailableError`` naming its absence."""
        try:
            return self._get_master_key()
        except ValueError as exc:
            raise DecryptionKeyUnavailableError(
                "the export is encrypted and no KESTREL_DATA_KEY is configured"
            ) from exc

    def _encrypt_content(self, content: bytes,
                         content_hash: Optional[str] = None) -> Tuple[bytes, str]:
        """Encrypt content with a portable, deterministically-derived key.

        The per-content key is derived from the master key + the content hash
        (F187): retrieval re-derives the identical key from ``KESTREL_DATA_KEY``
        + ``key_hash`` alone, so an encrypted CID is restorable on a different
        host WITHOUT a local key sidecar. ``key_hash`` returned here IS the
        content hash and is the only out-of-band material that must accompany
        the CID.
        """
        from kestrel_sdk.security.aead import AEADCipher

        if content_hash is None:
            content_hash = hashlib.sha256(content).hexdigest()

        master_key = self._get_master_key()
        content_key = self._derive_content_key(master_key, content_hash)
        encrypted_content = AEADCipher(content_key).encrypt(content)

        # The content hash doubles as the key identifier. No sidecar is
        # written — the key is reproducible from (master_key, content_hash).
        return encrypted_content, content_hash

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
    
    def _store_local_cache(self, content_hash: str, content: bytes,
                           metadata: Optional[Dict] = None, *,
                           tier=None, cid: Optional[str] = None):
        """Store content in local cache.

        ``tier``/``cid`` record DURABILITY in the ``.meta`` so ``cleanup_cache``
        can tell whether a remote replica exists (#1725) — LOCAL_ONLY content
        lives ONLY here and must never be auto-evicted.
        """
        cache_file = self.cache_dir / f"{content_hash}.cache"
        with open(cache_file, 'wb') as f:
            f.write(content)

        # Always write a .meta carrying the durability marker (reserved
        # ``_kestrel_*`` keys) so cleanup is tier-aware even when the caller
        # passed no user metadata.
        meta: Dict[str, Any] = dict(metadata or {})
        if tier is not None:
            meta["_kestrel_tier"] = tier.value if hasattr(tier, "value") else str(tier)
        if cid:
            meta["_kestrel_cid"] = cid
        meta_file = self.cache_dir / f"{content_hash}.meta"
        with open(meta_file, 'w') as f:
            json.dump(meta, f)
    
    def _retrieve_local_cache(
        self,
        content_hash: str,
        *,
        max_bytes: Optional[int] = None,
    ) -> Optional[bytes]:
        """Retrieve content from local cache"""
        cache_file = self.cache_dir / f"{content_hash}.cache"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                content = f.read() if max_bytes is None else f.read(max_bytes + 1)
            if max_bytes is not None and len(content) > max_bytes:
                raise ContentRetrievalLimitError(
                    "cached content exceeds configured input limit"
                )
            return content
        return None
    
    def _store_ipfs(
        self,
        content: bytes,
        metadata: Optional[Dict] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        *,
        filename: Optional[str] = None,
    ) -> str:
        """Store content in IPFS and return CID"""
        try:
            # Add content to IPFS
            file_content = (
                _ProgressBytesIO(content, on_progress) if on_progress else content
            )
            files = {'file': (filename or _DEFAULT_UPLOAD_NAME, file_content)}
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
    
    def _retrieve_ipfs(
        self,
        cid: str,
        *,
        max_bytes: Optional[int] = None,
    ) -> bytes:
        """Retrieve content from IPFS by CID"""
        try:
            response = requests.post(
                f"{self.ipfs_api_url}/api/v0/cat",
                params={'arg': cid},
                timeout=HTTP_TIMEOUT_DEFAULT,
                stream=max_bytes is not None,
            )
            
            if response.status_code == 200:
                if max_bytes is None:
                    return response.content
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        parsed_length = int(content_length)
                    except (TypeError, ValueError):
                        parsed_length = None
                    if parsed_length is not None and parsed_length > max_bytes:
                        response.close()
                        raise ContentRetrievalLimitError(
                            "IPFS response exceeds configured input limit"
                        )
                chunks: list[bytes] = []
                total = 0
                try:
                    for chunk in response.iter_content(
                        chunk_size=_RETRIEVAL_CHUNK_BYTES
                    ):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > max_bytes:
                            raise ContentRetrievalLimitError(
                                "IPFS response exceeds configured input limit"
                            )
                        chunks.append(chunk)
                finally:
                    response.close()
                return b''.join(chunks)
            else:
                if max_bytes is not None:
                    response.close()
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
        """Evict old cache files THAT HAVE A DURABLE REMOTE REPLICA.

        TIER-AWARE (#1725): the cache is the ONLY copy of ``LOCAL_ONLY`` content,
        so it must never be auto-deleted here — the old age-only sweep destroyed
        local-only data permanently. We only evict content whose ``.meta`` records
        a remote tier (IPFS/Filecoin) with a CID. Content with no/unknown
        durability marker (legacy entries, local-only) is KEPT (fail safe).
        Per-content key files (``key_*.key``) are never globbed here.
        """
        import time
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        durable_tiers = {
            StorageTier.IPFS.value,
            StorageTier.FILECOIN.value,
            StorageTier.ENCRYPTED_FILECOIN.value,
        }

        cleaned = 0
        for cache_file in self.cache_dir.glob('*.cache'):
            if cache_file.stat().st_mtime >= cutoff_time:
                continue
            meta_file = cache_file.with_suffix('.meta')
            tier = None
            cid = None
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    tier = meta.get("_kestrel_tier")
                    cid = meta.get("_kestrel_cid")
                except (OSError, ValueError):
                    pass
            # Keep anything we can't CONFIRM is durably replicated elsewhere.
            if tier not in durable_tiers or not cid:
                logging.debug(
                    "Keeping cache %s (tier=%s, cid=%s): no confirmed remote replica.",
                    cache_file.name, tier, bool(cid),
                )
                continue
            cache_file.unlink()
            if meta_file.exists():
                meta_file.unlink()
            cleaned += 1

        logging.info(f"🧹 Cleaned {cleaned} old cache files")
