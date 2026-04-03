"""
UCAN v1 Invocation Builder for Storacha (web3.storage)

Implements the w3up protocol for Python server-side use:
- CIDv1 computation (dag-cbor and raw codecs, sha2-256)
- CARv1 build/parse (for invocation transport and proof inclusion)
- UCAN v1 invocation signing (Ed25519 over DAG-CBOR encoded blocks)

One-time setup using the w3 CLI:
    npm install -g @web3-storage/w3cli
    w3 key create                                         # → STORACHA_AGENT_KEY
    w3 space create kestrel                               # → STORACHA_SPACE_DID
    w3 delegation create --can '*' <agent-did> | base64  # → STORACHA_PROOF

UCAN v1 spec:        https://github.com/ucan-wg/spec
Storacha bridge API: https://docs.storacha.network/
CAR v1 spec:         https://ipld.io/specs/transport/car/carv1/
"""

import base64
import hashlib
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    import cbor2
except ImportError:
    cbor2 = None  # type: ignore[assignment]
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STORACHA_SERVICE_DID = "did:web:up.storacha.network"
STORACHA_BRIDGE_URL = "https://up.storacha.network"
UCAN_VERSION = "1.0.0-rc.1"

# Multicodec values
_MC_DAG_CBOR = 0x71   # dag-cbor
_MC_RAW = 0x55        # raw bytes
_MC_SHA2_256 = 0x12   # sha2-256 hash function
_MC_ED25519_PUB = 0xED   # ed25519-pub (varint: 0xED → [0xED, 0x01])
_MC_ED25519_PRIV = 0x1300  # ed25519-priv (varint: 0x1300 → [0x80, 0x26])

# Base58btc alphabet (used for did:key multibase encoding)
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

# ---------------------------------------------------------------------------
# Varint utilities (LEB128 unsigned)
# ---------------------------------------------------------------------------

def _encode_varint(n: int) -> bytes:
    """Encode a non-negative integer as unsigned LEB128 varint."""
    if n == 0:
        return b"\x00"
    out = []
    while n:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
    return bytes(out)


def _read_varint(data: bytes, offset: int) -> Tuple[int, int]:
    """
    Decode a LEB128 varint starting at `offset`.

    Returns:
        (value, bytes_consumed)
    """
    value = 0
    shift = 0
    consumed = 0
    for _ in range(9):  # max 9 bytes for a 64-bit value
        if offset + consumed >= len(data):
            raise ValueError(f"Truncated varint at offset {offset}")
        byte = data[offset + consumed]
        consumed += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            break
    return value, consumed


def _cid_byte_length(data: bytes, offset: int = 0) -> int:
    """Return the byte length of a CIDv1 starting at `offset` in `data`."""
    start = offset
    _, n = _read_varint(data, offset); offset += n  # version
    _, n = _read_varint(data, offset); offset += n  # codec
    _, n = _read_varint(data, offset); offset += n  # hash function
    digest_len, n = _read_varint(data, offset); offset += n  # digest size
    offset += digest_len  # skip digest bytes
    return offset - start


# ---------------------------------------------------------------------------
# CID computation
# ---------------------------------------------------------------------------

def cid_v1(data: bytes, codec: int = _MC_DAG_CBOR) -> bytes:
    """
    Compute a CIDv1 byte string for `data` using sha2-256.

    Args:
        data:  Raw bytes to hash
        codec: Multicodec code for the content type (default dag-cbor 0x71)

    Returns:
        36-byte CIDv1 (1-byte version + 1-byte codec + 34-byte sha2-256 multihash)
    """
    digest = hashlib.sha256(data).digest()
    multihash = _encode_varint(_MC_SHA2_256) + _encode_varint(len(digest)) + digest
    return _encode_varint(1) + _encode_varint(codec) + multihash


def cid_to_string(cid_bytes: bytes) -> str:
    """
    Encode CID bytes as a base32lower CIDv1 string (multibase prefix "b").

    Example output: "bafkreihu..."
    """
    encoded = base64.b32encode(cid_bytes).decode().lower().rstrip("=")
    return "b" + encoded


# ---------------------------------------------------------------------------
# Base58btc (for did:key DID encoding)
# ---------------------------------------------------------------------------

def _base58btc_encode(data: bytes) -> str:
    """Encode bytes as a base58btc string (no multibase prefix)."""
    n = int.from_bytes(data, "big")
    result = []
    while n:
        n, rem = divmod(n, 58)
        result.append(_B58_ALPHABET[rem])
    # Preserve leading zero bytes as '1'
    for byte in data:
        if byte == 0:
            result.append(_B58_ALPHABET[0])
        else:
            break
    return "".join(reversed(result))


# ---------------------------------------------------------------------------
# CAR v1 build/parse
# ---------------------------------------------------------------------------

def build_car(root_cids: List[bytes], blocks: List[Tuple[bytes, bytes]]) -> bytes:
    """
    Build a CARv1 binary.

    Args:
        root_cids: List of raw CID byte strings for the root blocks.
        blocks:    List of (cid_bytes, block_data) pairs in write order.

    Returns:
        CARv1 bytes ready to POST to the Storacha bridge.
    """
    header = {
        "version": 1,
        "roots": [cbor2.CBORTag(42, b"\x00" + cid) for cid in root_cids],
    }
    header_bytes = cbor2.dumps(header, canonical=True)

    out = _encode_varint(len(header_bytes)) + header_bytes
    for cid_bytes, block_data in blocks:
        section = cid_bytes + block_data
        out += _encode_varint(len(section)) + section
    return out


def parse_car(car_bytes: bytes) -> Tuple[List[bytes], Dict[bytes, bytes]]:
    """
    Parse a CARv1 binary.

    Returns:
        (root_cid_list, {cid_bytes: block_data})

    Note: CID bytes in the returned dict use raw CID bytes (no multibase prefix).
    """
    offset = 0

    header_len, n = _read_varint(car_bytes, offset)
    offset += n
    header = cbor2.loads(car_bytes[offset: offset + header_len])
    offset += header_len

    root_cids: List[bytes] = []
    for r in header.get("roots", []):
        if isinstance(r, cbor2.CBORTag) and r.tag == 42:
            root_cids.append(r.value[1:])  # strip multibase identity prefix (0x00)

    blocks: Dict[bytes, bytes] = {}
    while offset < len(car_bytes):
        section_len, n = _read_varint(car_bytes, offset)
        offset += n
        section = car_bytes[offset: offset + section_len]
        offset += section_len

        cid_len = _cid_byte_length(section)
        cid = section[:cid_len]
        data = section[cid_len:]
        blocks[cid] = data

    return root_cids, blocks


# ---------------------------------------------------------------------------
# StorachaUCAN – key loading, DID derivation, invocation signing
# ---------------------------------------------------------------------------

class StorachaUCAN:
    """
    UCAN v1 invocation builder for the Storacha w3up protocol.

    Handles Ed25519 key loading, DID derivation, DAG-CBOR invocation encoding,
    and CARv1 wrapping for submission to the Storacha bridge.

    Environment variables consumed:
        STORACHA_AGENT_KEY   Ed25519 private key (from `w3 key create`)
        STORACHA_SPACE_DID   Space DID (from `w3 space create`)
        STORACHA_PROOF       Base64-encoded delegation proof CAR

    Key format:
        `w3 key create` produces a multibase base64pad string starting with "M":
            M = multibase base64pad prefix
            decoded bytes = [0x80, 0x26] (ed25519-priv varint) + 32-byte seed

        Raw base64-encoded 32-byte seeds (without multicodec prefix) are also
        accepted as a convenience for testing.
    """

    def __init__(self, agent_key: str, space_did: str, proof: str):
        """
        Args:
            agent_key: Ed25519 key string (multibase "M..." or raw base64)
            space_did: Space DID (e.g. "did:key:z6Mk...")
            proof:     Base64-encoded CARv1 bytes of the UCAN delegation
        """
        if cbor2 is None:
            raise ImportError(
                "cbor2 package is required for StorachaUCAN. "
                "Install it with: pip install kestrel-sovereign[wallet]"
            )
        self._space_did = space_did
        self._private_key = self._load_agent_key(agent_key)

        pub_bytes = self._private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )
        self._agent_did = _pubkey_to_did(pub_bytes)

        # Parse the delegation proof CAR once at startup
        proof_bytes = base64.b64decode(_pad_base64(proof))
        self._proof_root_cids, self._proof_blocks = parse_car(proof_bytes)

        if not self._proof_root_cids:
            raise ValueError("Delegation proof CAR contains no root CIDs")

        logger.debug("StorachaUCAN ready — agent: %s", self._agent_did)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def agent_did(self) -> str:
        """The DID of the agent key (used as UCAN issuer)."""
        return self._agent_did

    @property
    def space_did(self) -> str:
        """The space DID (used as UCAN resource)."""
        return self._space_did

    @property
    def proof_root_cid(self) -> bytes:
        """Raw CID bytes of the delegation proof root block."""
        return self._proof_root_cids[0]

    # ------------------------------------------------------------------
    # Invocation builders
    # ------------------------------------------------------------------

    def build_store_add(
        self,
        content_cid: bytes,
        size: int,
        exp: Optional[int] = None,
    ) -> Tuple[bytes, bytes]:
        """
        Build a signed store/add UCAN invocation.

        Args:
            content_cid: Raw CID bytes of the content/CAR shard to store
            size:        Byte size of the CAR shard

        Returns:
            (signed_block_bytes, cid_bytes_of_signed_block)
        """
        nb: Dict[str, Any] = {
            "link": cbor2.CBORTag(42, b"\x00" + content_cid),
            "size": size,
        }
        return self._sign_invocation("store/add", nb, exp=exp)

    def build_upload_add(
        self,
        root_cid: bytes,
        shard_cids: Optional[List[bytes]] = None,
        exp: Optional[int] = None,
    ) -> Tuple[bytes, bytes]:
        """
        Build a signed upload/add UCAN invocation.

        Args:
            root_cid:   Raw CID bytes of the content root
            shard_cids: Raw CID bytes of each CAR shard (may be empty)

        Returns:
            (signed_block_bytes, cid_bytes_of_signed_block)
        """
        nb: Dict[str, Any] = {
            "root": cbor2.CBORTag(42, b"\x00" + root_cid),
            "shards": [
                cbor2.CBORTag(42, b"\x00" + s) for s in (shard_cids or [])
            ],
        }
        return self._sign_invocation("upload/add", nb, exp=exp)

    def build_invocation_car(
        self,
        invocations: List[Tuple[bytes, bytes]],
    ) -> bytes:
        """
        Wrap one or more signed invocations in a CARv1 for bridge submission.

        Includes all delegation proof blocks so the bridge can verify the chain.

        Args:
            invocations: List of (block_bytes, cid_bytes) from build_* methods.
                         The last entry's CID becomes the CAR root.

        Returns:
            CARv1 bytes ready to POST to STORACHA_BRIDGE_URL.
        """
        _, last_cid = invocations[-1]
        root_cids = [last_cid]

        # proof blocks first, then invocation blocks
        blocks: List[Tuple[bytes, bytes]] = list(self._proof_blocks.items())
        for block_bytes, cid_bytes in invocations:
            blocks.append((cid_bytes, block_bytes))

        return build_car(root_cids, blocks)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def content_cid(content: bytes) -> bytes:
        """Compute CIDv1 for raw content bytes (raw codec, sha2-256)."""
        return cid_v1(content, _MC_RAW)

    @staticmethod
    def cid_to_str(cid_bytes: bytes) -> str:
        """Convert raw CID bytes to a base32lower multibase string ("b...")."""
        return cid_to_string(cid_bytes)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sign_invocation(
        self,
        can: str,
        nb: Dict[str, Any],
        exp: Optional[int] = None,
    ) -> Tuple[bytes, bytes]:
        """
        Build, encode, and sign a UCAN v1 invocation.

        Signing protocol (UCAN v1):
        1. Build unsigned invocation dict (without "s" field)
        2. DAG-CBOR encode with canonical key ordering
        3. Compute CIDv1 of the encoded bytes (dag-cbor codec)
        4. Sign the raw CID bytes with Ed25519
        5. Add signature as bytes in "s" field; re-encode to get final block

        Returns:
            (signed_block_bytes, cid_of_signed_block)
        """
        if exp is None:
            exp = int(time.time()) + 3600  # valid for 1 hour

        unsigned: Dict[str, Any] = {
            "att": [{"can": can, "nb": nb, "with": self._space_did}],
            "aud": STORACHA_SERVICE_DID,
            "exp": exp,
            "iss": self._agent_did,
            "nce": str(uuid.uuid4()),
            "prf": [cbor2.CBORTag(42, b"\x00" + self.proof_root_cid)],
            "v": UCAN_VERSION,
        }

        # DAG-CBOR encode with CBOR canonical ordering (same as IPLD dag-cbor
        # for ASCII string keys: length-first, then lexicographic)
        unsigned_bytes = cbor2.dumps(unsigned, canonical=True)

        # Signing input = raw bytes of CIDv1(dag-cbor, sha256(unsigned_bytes))
        unsigned_cid = cid_v1(unsigned_bytes, _MC_DAG_CBOR)
        signature: bytes = self._private_key.sign(unsigned_cid)

        signed: Dict[str, Any] = {**unsigned, "s": signature}
        signed_bytes = cbor2.dumps(signed, canonical=True)
        signed_cid = cid_v1(signed_bytes, _MC_DAG_CBOR)

        return signed_bytes, signed_cid

    @staticmethod
    def _load_agent_key(encoded: str) -> Ed25519PrivateKey:
        """
        Parse an Ed25519 private key from encoded string.

        Accepts:
        - "M..." multibase base64pad with ed25519-priv multicodec prefix
          (output of `w3 key create`)
        - Raw base64url or base64 of 32-byte seed (for testing)
        """
        encoded = encoded.strip()
        if encoded.startswith("M"):
            # Could be multibase base64pad (w3 key create) OR a raw base64
            # string that happens to start with "M".  Disambiguate by checking
            # for the ed25519-priv multicodec prefix after decoding.
            raw = base64.b64decode(_pad_base64(encoded[1:]))
            if len(raw) >= 2 and raw[0] == 0x80 and raw[1] == 0x26:
                # Genuine multibase: strip the multicodec varint prefix
                _, n = _read_varint(raw, 0)
                seed = raw[n:]
            else:
                # Not a real multicodec header — treat the full string as
                # raw base64-encoded 32-byte seed.
                seed = base64.b64decode(_pad_base64(encoded))
        else:
            # Treat as raw base64-encoded 32-byte seed
            seed = base64.b64decode(_pad_base64(encoded))

        if len(seed) != 32:
            raise ValueError(
                f"Expected a 32-byte Ed25519 seed, got {len(seed)} bytes. "
                "Use the key produced by `w3 key create` for STORACHA_AGENT_KEY."
            )
        return Ed25519PrivateKey.from_private_bytes(seed)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _pubkey_to_did(pub_bytes: bytes) -> str:
    """
    Derive a did:key DID from 32 raw Ed25519 public key bytes.

    Format: "did:key:z" + base58btc(varint(0xED) + pubkey_bytes)
    """
    # ed25519-pub multicodec = 0xED (varint → [0xED, 0x01])
    prefixed = _encode_varint(_MC_ED25519_PUB) + pub_bytes
    return "did:key:z" + _base58btc_encode(prefixed)


def _pad_base64(s: str) -> str:
    """Add padding to base64 string if needed."""
    missing = len(s) % 4
    return s + "=" * (4 - missing) if missing else s
