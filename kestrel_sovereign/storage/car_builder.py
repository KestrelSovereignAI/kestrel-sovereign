"""
CAR v1 (Content Addressable aRchive) builder and reader.

Implements the CAR v1 spec for packing sovereignty exports into a single
verifiable content-addressed archive. One upload, one CID, full DAG.

CAR v1 format:
  [header]  = varint(len(header_bytes)) + dag-cbor({"version": 1, "roots": [cid]})
  [block]*  = varint(len(cid_bytes + data)) + cid_bytes + data

CID v1 format:
  bytes: 0x01 + varint(codec) + multihash
  string: 'b' + base32lower(cid_bytes)   (multibase prefix 'b')

Codecs:
  0x55 = raw
  0x71 = dag-cbor

Multihash (sha2-256):
  varint(0x12) + varint(32) + sha256_digest

References:
  - CAR spec: https://ipld.io/specs/transport/car/carv1/
  - CID spec: https://github.com/multiformats/cid
  - Multihash: https://github.com/multiformats/multihash
"""

import base64
import hashlib
import io
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Codec constants
CODEC_RAW = 0x55
CODEC_DAG_CBOR = 0x71

# Hash function constants
HASH_SHA2_256 = 0x12
HASH_SHA2_256_LENGTH = 32

# CID version
CID_VERSION_1 = 0x01

# CBOR tag for CID links in dag-cbor
CBOR_TAG_CID = 42


# =============================================================================
# Varint (unsigned LEB128)
# =============================================================================


def encode_varint(n: int) -> bytes:
    """Encode an unsigned integer as a varint (unsigned LEB128)."""
    if n < 0:
        raise ValueError("Varint must be non-negative")
    parts = []
    while n >= 0x80:
        parts.append((n & 0x7F) | 0x80)
        n >>= 7
    parts.append(n & 0x7F)
    return bytes(parts)


def decode_varint(data: bytes, offset: int = 0) -> Tuple[int, int]:
    """
    Decode a varint from bytes at the given offset.

    Returns:
        (value, new_offset) tuple
    """
    result = 0
    shift = 0
    pos = offset
    while pos < len(data):
        byte = data[pos]
        result |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("Varint too long")
    raise ValueError("Unexpected end of data while decoding varint")


# =============================================================================
# Base32 (RFC 4648, lowercase, no padding)
# =============================================================================


_B32_ALPHABET = "abcdefghijklmnopqrstuvwxyz234567"
_B32_DECODE = {c: i for i, c in enumerate(_B32_ALPHABET)}


def _base32_encode(data: bytes) -> str:
    """Encode bytes to base32lower (RFC 4648, no padding)."""
    # Use stdlib base32 then lowercase and strip padding
    encoded = base64.b32encode(data).decode("ascii").lower().rstrip("=")
    return encoded


def _base32_decode(s: str) -> bytes:
    """Decode base32lower string to bytes."""
    # Add padding back for stdlib
    padding = (8 - len(s) % 8) % 8
    padded = s.upper() + "=" * padding
    return base64.b32decode(padded)


# =============================================================================
# Multihash
# =============================================================================


def _build_multihash(data: bytes) -> bytes:
    """Build a sha2-256 multihash for the given data."""
    digest = hashlib.sha256(data).digest()
    return encode_varint(HASH_SHA2_256) + encode_varint(HASH_SHA2_256_LENGTH) + digest


def _verify_multihash(multihash: bytes, data: bytes) -> bool:
    """Verify that data matches a sha2-256 multihash."""
    hash_code, pos = decode_varint(multihash, 0)
    if hash_code != HASH_SHA2_256:
        raise ValueError(f"Unsupported hash function: 0x{hash_code:02x}")
    digest_len, pos = decode_varint(multihash, pos)
    if digest_len != HASH_SHA2_256_LENGTH:
        raise ValueError(f"Unexpected digest length: {digest_len}")
    expected_digest = multihash[pos : pos + digest_len]
    actual_digest = hashlib.sha256(data).digest()
    return expected_digest == actual_digest


# =============================================================================
# CID v1
# =============================================================================


def build_cid_bytes(codec: int, data: bytes) -> bytes:
    """Build CID v1 bytes for the given codec and data."""
    multihash = _build_multihash(data)
    return bytes([CID_VERSION_1]) + encode_varint(codec) + multihash


def cid_bytes_to_string(cid_bytes: bytes) -> str:
    """Convert CID bytes to a multibase-encoded string (base32lower, prefix 'b')."""
    return "b" + _base32_encode(cid_bytes)


def cid_string_to_bytes(cid_str: str) -> bytes:
    """Convert a multibase CID string back to bytes."""
    if not cid_str.startswith("b"):
        raise ValueError(f"Unsupported multibase prefix: {cid_str[0]!r} (expected 'b')")
    return _base32_decode(cid_str[1:])


def compute_raw_cid(data: bytes) -> Tuple[bytes, str]:
    """Compute CID v1 for raw data. Returns (cid_bytes, cid_string)."""
    cid_bytes = build_cid_bytes(CODEC_RAW, data)
    return cid_bytes, cid_bytes_to_string(cid_bytes)


def compute_dag_cbor_cid(cbor_data: bytes) -> Tuple[bytes, str]:
    """Compute CID v1 for dag-cbor data. Returns (cid_bytes, cid_string)."""
    cid_bytes = build_cid_bytes(CODEC_DAG_CBOR, cbor_data)
    return cid_bytes, cid_bytes_to_string(cid_bytes)


def _parse_cid_from_bytes(data: bytes, offset: int) -> Tuple[bytes, int]:
    """
    Parse a CID from a byte stream at the given offset.

    Returns:
        (cid_bytes, new_offset)
    """
    start = offset

    # CID version
    version, offset = decode_varint(data, offset)
    if version != CID_VERSION_1:
        raise ValueError(f"Unsupported CID version: {version}")

    # Codec
    _codec, offset = decode_varint(data, offset)

    # Multihash: hash_code + digest_length + digest
    _hash_code, offset = decode_varint(data, offset)
    digest_len, offset = decode_varint(data, offset)
    offset += digest_len

    return data[start:offset], offset


# =============================================================================
# dag-cbor encoding (minimal, using cbor2)
# =============================================================================


def _dag_cbor_encode(obj: dict) -> bytes:
    """Encode a dict as dag-cbor.

    CID references in the dict should be bytes objects tagged with CBOR tag 42
    (prefix 0x00 + cid_bytes). We handle this via cbor2's tag support.
    """
    try:
        import cbor2
    except ImportError:
        raise ImportError(
            "cbor2 is required for CAR file support. Install with: pip install cbor2"
        )
    return cbor2.dumps(obj, canonical=True)


def _dag_cbor_decode(data: bytes) -> dict:
    """Decode dag-cbor bytes to a dict."""
    try:
        import cbor2
    except ImportError:
        raise ImportError(
            "cbor2 is required for CAR file support. Install with: pip install cbor2"
        )
    return cbor2.loads(data)


def make_cid_link(cid_bytes: bytes) -> object:
    """Create a dag-cbor CID link (CBOR tag 42 with 0x00 prefix)."""
    try:
        import cbor2
    except ImportError:
        raise ImportError("cbor2 is required for CAR file support.")
    # dag-cbor CID links are Tag 42 wrapping bytes(0x00 + cid_bytes)
    return cbor2.CBORTag(CBOR_TAG_CID, b"\x00" + cid_bytes)


# =============================================================================
# CARBuilder
# =============================================================================


class CARBuilder:
    """
    Build CAR v1 archives for sovereignty exports.

    Usage:
        builder = CARBuilder()
        cid1 = builder.add_raw_block(shard1_bytes)
        cid2 = builder.add_raw_block(shard2_bytes)
        manifest_cid = builder.add_dag_cbor_block({"shards": [cid1, cid2]})
        builder.set_root(manifest_cid)
        car_bytes = builder.build()
    """

    def __init__(self):
        self._blocks: List[Tuple[bytes, bytes]] = []  # (cid_bytes, data)
        self._root_cid_bytes: Optional[bytes] = None
        self._cid_index: Dict[str, int] = {}  # cid_string -> block index

    def add_raw_block(self, data: bytes) -> str:
        """
        Add a raw data block.

        Args:
            data: Raw block content

        Returns:
            CID string (base32-encoded)
        """
        cid_bytes, cid_str = compute_raw_cid(data)
        if cid_str not in self._cid_index:
            self._cid_index[cid_str] = len(self._blocks)
            self._blocks.append((cid_bytes, data))
        return cid_str

    def add_dag_cbor_block(self, obj: dict) -> str:
        """
        Add a dag-cbor encoded block (typically the manifest).

        Args:
            obj: Dict to encode as dag-cbor

        Returns:
            CID string (base32-encoded)
        """
        cbor_data = _dag_cbor_encode(obj)
        cid_bytes, cid_str = compute_dag_cbor_cid(cbor_data)
        if cid_str not in self._cid_index:
            self._cid_index[cid_str] = len(self._blocks)
            self._blocks.append((cid_bytes, cbor_data))
        return cid_str

    def set_root(self, cid_str: str) -> None:
        """
        Set the root CID of the CAR archive.

        Args:
            cid_str: CID string of the root block (must have been added)
        """
        if cid_str not in self._cid_index:
            raise ValueError(f"CID not found in builder: {cid_str}")
        self._root_cid_bytes = cid_string_to_bytes(cid_str)

    def build(self) -> bytes:
        """
        Build the complete CAR v1 archive.

        Returns:
            CAR v1 file bytes

        Raises:
            ValueError: If no root CID has been set
        """
        if self._root_cid_bytes is None:
            raise ValueError("Root CID must be set before building")
        if not self._blocks:
            raise ValueError("No blocks to write")

        buf = io.BytesIO()

        # Header: dag-cbor({"version": 1, "roots": [root_cid_link]})
        root_link = make_cid_link(self._root_cid_bytes)
        header_obj = {"version": 1, "roots": [root_link]}
        header_bytes = _dag_cbor_encode(header_obj)

        # Write header with length prefix
        buf.write(encode_varint(len(header_bytes)))
        buf.write(header_bytes)

        # Write blocks: varint(len(cid + data)) + cid + data
        for cid_bytes, data in self._blocks:
            block_payload = cid_bytes + data
            buf.write(encode_varint(len(block_payload)))
            buf.write(block_payload)

        return buf.getvalue()

    @property
    def block_count(self) -> int:
        """Number of blocks in the builder."""
        return len(self._blocks)


# =============================================================================
# CARReader
# =============================================================================


class CARReader:
    """
    Read and verify CAR v1 archives.

    Usage:
        reader = CARReader(car_bytes)
        print(reader.root_cid)
        data = reader.get_block(some_cid)
        assert reader.verify()
    """

    def __init__(self, car_bytes: bytes):
        """
        Parse a CAR v1 archive from bytes.

        Args:
            car_bytes: Complete CAR v1 file bytes
        """
        self._blocks: Dict[str, bytes] = {}  # cid_string -> data
        self._cid_bytes_map: Dict[str, bytes] = {}  # cid_string -> cid_bytes
        self._root_cid_str: str = ""
        self._parse(car_bytes)

    def _parse(self, car_bytes: bytes) -> None:
        """Parse the CAR v1 format."""
        offset = 0

        # Read header
        header_len, offset = decode_varint(car_bytes, offset)
        header_data = car_bytes[offset : offset + header_len]
        offset += header_len

        header = _dag_cbor_decode(header_data)
        if header.get("version") != 1:
            raise ValueError(f"Unsupported CAR version: {header.get('version')}")

        roots = header.get("roots", [])
        if not roots:
            raise ValueError("CAR file has no roots")

        # Extract root CID from dag-cbor tag 42
        root_ref = roots[0]
        try:
            import cbor2
        except ImportError:
            raise ImportError("cbor2 is required for CAR file support.")

        if isinstance(root_ref, cbor2.CBORTag) and root_ref.tag == CBOR_TAG_CID:
            # Strip the 0x00 prefix from the CID link
            root_cid_bytes = root_ref.value[1:]  # Remove identity multibase prefix
        elif isinstance(root_ref, bytes):
            root_cid_bytes = root_ref
        else:
            raise ValueError(f"Unexpected root CID type: {type(root_ref)}")

        self._root_cid_str = cid_bytes_to_string(root_cid_bytes)

        # Read blocks
        while offset < len(car_bytes):
            block_len, offset = decode_varint(car_bytes, offset)
            block_data = car_bytes[offset : offset + block_len]
            offset += block_len

            # Parse CID from the start of the block
            cid_bytes, cid_end = _parse_cid_from_bytes(block_data, 0)
            data = block_data[cid_end:]

            cid_str = cid_bytes_to_string(cid_bytes)
            self._blocks[cid_str] = data
            self._cid_bytes_map[cid_str] = cid_bytes

    @property
    def root_cid(self) -> str:
        """Get the root CID string."""
        return self._root_cid_str

    def get_block(self, cid: str) -> Optional[bytes]:
        """
        Get raw block data by CID string.

        Args:
            cid: CID string (base32-encoded)

        Returns:
            Block data bytes, or None if not found
        """
        return self._blocks.get(cid)

    def get_dag_cbor_block(self, cid: str) -> Optional[dict]:
        """
        Get and decode a dag-cbor block.

        Args:
            cid: CID string

        Returns:
            Decoded dict, or None if not found
        """
        data = self._blocks.get(cid)
        if data is None:
            return None
        return _dag_cbor_decode(data)

    def list_cids(self) -> List[str]:
        """List all CIDs in the archive."""
        return list(self._blocks.keys())

    def verify(self) -> bool:
        """
        Verify all blocks match their CIDs.

        Returns:
            True if all blocks are valid
        """
        for cid_str, data in self._blocks.items():
            cid_bytes = self._cid_bytes_map[cid_str]

            # Parse codec from CID
            offset = 0
            _version, offset = decode_varint(cid_bytes, offset)
            codec, offset = decode_varint(cid_bytes, offset)

            # Extract multihash
            multihash = cid_bytes[offset:]

            if not _verify_multihash(multihash, data):
                logger.error(f"Block verification failed for CID: {cid_str}")
                return False

        return True

    @property
    def block_count(self) -> int:
        """Number of blocks in the archive."""
        return len(self._blocks)
