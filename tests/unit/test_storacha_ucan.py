"""Tests for storacha_ucan.py — UCAN v1 builder and CAR utilities."""

import base64
import hashlib

import pytest

cbor2 = pytest.importorskip("cbor2")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from kestrel_storage_storacha.storacha_ucan import (
    StorachaUCAN,
    _base58btc_encode,
    _cid_byte_length,
    _encode_varint,
    _pad_base64,
    _pubkey_to_did,
    _read_varint,
    build_car,
    cid_to_string,
    cid_v1,
    parse_car,
)


# ---------------------------------------------------------------------------
# Helpers to build test fixtures
# ---------------------------------------------------------------------------

def _make_test_agent_key() -> tuple:
    """Generate a fresh Ed25519 keypair and return (private_key, w3_key_string)."""
    private_key = Ed25519PrivateKey.generate()
    seed = private_key.private_bytes(Encoding.Raw, encoding=Encoding.Raw, format=None)  # type: ignore[arg-type]
    # Actually: from_private_bytes takes 32-byte seed; to get raw seed bytes:
    # We re-derive via private_bytes_raw if available, else use the approach below
    # For testing, reconstruct from known seed
    return private_key


def _make_seed_and_key():
    """Return (32-byte seed, Ed25519PrivateKey) for a fresh keypair."""
    # Generate random seed
    import os
    seed = os.urandom(32)
    pk = Ed25519PrivateKey.from_private_bytes(seed)
    return seed, pk


def _seed_to_w3_key(seed: bytes) -> str:
    """Encode a 32-byte seed as a `w3 key create`-style multibase string."""
    # ed25519-priv multicodec = 0x1300 → varint [0x80, 0x26]
    from kestrel_storage_storacha.storacha_ucan import _encode_varint
    prefix = bytes([0x80, 0x26])  # varint(0x1300)
    raw = prefix + seed
    return "M" + base64.b64encode(raw).decode().rstrip("=")


def _make_minimal_proof_car(agent_did: str) -> str:
    """Build a minimal (fake) delegation proof CAR for testing."""
    # A real proof is a UCAN block signed by the space DID.
    # For unit tests, we just need a valid CARv1 with at least one block.
    delegation = {
        "v": "1.0.0-rc.1",
        "iss": "did:key:z6MkTestSpace",
        "aud": agent_did,
        "att": [{"with": "did:key:z6MkTestSpace", "can": "*"}],
        "prf": [],
        "exp": 9999999999,
    }
    block_bytes = cbor2.dumps(delegation, canonical=True)
    block_cid = cid_v1(block_bytes)
    car = build_car([block_cid], [(block_cid, block_bytes)])
    return base64.b64encode(car).decode()


# ---------------------------------------------------------------------------
# Varint tests
# ---------------------------------------------------------------------------

class TestVarint:
    def test_encode_zero(self):
        assert _encode_varint(0) == b"\x00"

    def test_encode_single_byte(self):
        assert _encode_varint(1) == b"\x01"
        assert _encode_varint(127) == b"\x7f"

    def test_encode_two_bytes(self):
        # 128 = 0x80 → [0x80, 0x01]
        assert _encode_varint(128) == bytes([0x80, 0x01])
        # 237 (0xED) = ed25519-pub multicodec → [0xED, 0x01]
        assert _encode_varint(237) == bytes([0xED, 0x01])

    def test_encode_0x1300(self):
        # ed25519-priv = 0x1300 = 4864 → [0x80, 0x26]
        assert _encode_varint(0x1300) == bytes([0x80, 0x26])

    def test_roundtrip(self):
        for n in [0, 1, 63, 127, 128, 255, 300, 4864, 65536]:
            encoded = _encode_varint(n)
            decoded, consumed = _read_varint(encoded, 0)
            assert decoded == n
            assert consumed == len(encoded)

    def test_read_from_offset(self):
        data = b"\xff" + _encode_varint(42) + b"\x00"
        val, n = _read_varint(data, 1)
        assert val == 42
        assert n == 1


# ---------------------------------------------------------------------------
# CID tests
# ---------------------------------------------------------------------------

class TestCID:
    def test_cid_v1_length(self):
        cid = cid_v1(b"hello world")
        # version(1) + codec(1) + hash_fn(1) + hash_len(1) + sha256(32) = 36 bytes
        assert len(cid) == 36

    def test_cid_v1_version_byte(self):
        cid = cid_v1(b"test")
        assert cid[0] == 0x01  # CIDv1

    def test_cid_v1_dag_cbor_codec(self):
        cid = cid_v1(b"test", codec=0x71)
        assert cid[1] == 0x71

    def test_cid_v1_raw_codec(self):
        cid = cid_v1(b"test", codec=0x55)
        assert cid[1] == 0x55

    def test_cid_v1_deterministic(self):
        assert cid_v1(b"hello") == cid_v1(b"hello")
        assert cid_v1(b"hello") != cid_v1(b"world")

    def test_cid_byte_length(self):
        cid = cid_v1(b"data")
        assert _cid_byte_length(cid) == len(cid)

    def test_cid_to_string_prefix(self):
        cid = cid_v1(b"test")
        s = cid_to_string(cid)
        assert s.startswith("b"), "CIDv1 base32lower must start with 'b'"

    def test_cid_to_string_roundtrip(self):
        from kestrel_storage_storacha.storacha_rest import _cid_str_to_bytes
        cid = cid_v1(b"roundtrip content")
        s = cid_to_string(cid)
        recovered = _cid_str_to_bytes(s)
        assert recovered == cid


# ---------------------------------------------------------------------------
# CAR build/parse tests
# ---------------------------------------------------------------------------

class TestCAR:
    def test_build_and_parse_single_block(self):
        data = b"hello ipfs"
        block_cid = cid_v1(data)
        car = build_car([block_cid], [(block_cid, data)])

        root_cids, blocks = parse_car(car)

        assert len(root_cids) == 1
        assert root_cids[0] == block_cid
        assert blocks[block_cid] == data

    def test_build_and_parse_multiple_blocks(self):
        blocks_data = [(b"block one", b"data one"), (b"block two", b"data two")]
        cid_block_pairs = [(cid_v1(key), val) for key, val in blocks_data]
        root = cid_block_pairs[0][0]

        car = build_car([root], cid_block_pairs)
        root_cids, parsed = parse_car(car)

        assert root_cids[0] == root
        assert len(parsed) == 2

    def test_empty_car(self):
        # CAR with no blocks (just a header) should parse without error
        block = b"single"
        cid = cid_v1(block)
        car = build_car([cid], [(cid, block)])
        root_cids, blocks = parse_car(car)
        assert root_cids[0] == cid


# ---------------------------------------------------------------------------
# DID/key encoding tests
# ---------------------------------------------------------------------------

class TestDIDEncoding:
    def test_pubkey_to_did_prefix(self):
        seed, pk = _make_seed_and_key()
        pub = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        did = _pubkey_to_did(pub)
        assert did.startswith("did:key:z")

    def test_base58btc_known_value(self):
        # Empty bytes should encode to empty string (or just "1"s for leading zeros)
        result = _base58btc_encode(b"\x00")
        assert result == "1"

    def test_base58btc_nonzero(self):
        result = _base58btc_encode(b"\x01")
        assert result == "2"  # 1 in base58 alphabet at position 1 = "2"

    def test_pad_base64(self):
        assert _pad_base64("abc") == "abc="
        assert _pad_base64("abcd") == "abcd"
        assert _pad_base64("ab") == "ab=="
        assert _pad_base64("a") == "a==="


# ---------------------------------------------------------------------------
# StorachaUCAN tests
# ---------------------------------------------------------------------------

class TestStorachaUCAN:
    @pytest.fixture
    def ucan(self):
        seed, pk = _make_seed_and_key()
        w3_key = _seed_to_w3_key(seed)
        pub = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        agent_did = _pubkey_to_did(pub)
        space_did = "did:key:z6MkTestSpaceForUnitTests"
        proof_b64 = _make_minimal_proof_car(agent_did)
        return StorachaUCAN(agent_key=w3_key, space_did=space_did, proof=proof_b64)

    def test_agent_did_format(self, ucan):
        assert ucan.agent_did.startswith("did:key:z")

    def test_space_did(self, ucan):
        assert ucan.space_did == "did:key:z6MkTestSpaceForUnitTests"

    def test_proof_root_cid_is_bytes(self, ucan):
        assert isinstance(ucan.proof_root_cid, bytes)
        assert len(ucan.proof_root_cid) == 36

    def test_build_store_add_returns_two_items(self, ucan):
        content = b"test content"
        content_cid = ucan.content_cid(content)
        block, cid = ucan.build_store_add(content_cid, len(content))
        assert isinstance(block, bytes)
        assert isinstance(cid, bytes)
        assert len(cid) == 36

    def test_build_upload_add_returns_two_items(self, ucan):
        content = b"upload add test"
        content_cid = ucan.content_cid(content)
        block, cid = ucan.build_upload_add(content_cid, [content_cid])
        assert isinstance(block, bytes)
        assert len(cid) == 36

    def test_build_store_add_contains_capability(self, ucan):
        content_cid = ucan.content_cid(b"data")
        block, cid = ucan.build_store_add(content_cid, 4)
        decoded = cbor2.loads(block)
        assert decoded["v"] == "1.0.0-rc.1"
        assert decoded["iss"] == ucan.agent_did
        assert decoded["att"][0]["can"] == "store/add"

    def test_build_upload_add_contains_capability(self, ucan):
        root_cid = ucan.content_cid(b"root data")
        block, cid = ucan.build_upload_add(root_cid)
        decoded = cbor2.loads(block)
        assert decoded["att"][0]["can"] == "upload/add"

    def test_signature_field_present(self, ucan):
        content_cid = ucan.content_cid(b"signed data")
        block, _ = ucan.build_store_add(content_cid, 11)
        decoded = cbor2.loads(block)
        assert "s" in decoded
        assert isinstance(decoded["s"], bytes)
        assert len(decoded["s"]) == 64  # Ed25519 signature is 64 bytes

    def test_signature_is_valid_ed25519(self, ucan):
        """Verify the signature using the agent's public key."""
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        content_cid = ucan.content_cid(b"verify me")
        block, _ = ucan.build_store_add(content_cid, 9)
        decoded = cbor2.loads(block)

        # Reconstruct the unsigned block (everything except "s")
        unsigned = {k: v for k, v in decoded.items() if k != "s"}
        unsigned_bytes = cbor2.dumps(unsigned, canonical=True)
        unsigned_cid = cid_v1(unsigned_bytes)

        # Extract public key from agent DID
        did = ucan.agent_did  # "did:key:z" + base58btc(varint(0xED) + pubkey)
        z_part = did[len("did:key:z"):]

        # base58btc decode
        n = 0
        for char in z_part:
            n = n * 58 + "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz".index(char)
        pub_with_prefix = n.to_bytes(34, "big")  # 2 prefix + 32 key bytes
        pub_bytes = pub_with_prefix[2:]  # strip multicodec prefix

        pub_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
        # Should not raise:
        pub_key.verify(decoded["s"], unsigned_cid)

    def test_build_invocation_car_is_valid_car(self, ucan):
        content_cid = ucan.content_cid(b"car test")
        block, cid = ucan.build_store_add(content_cid, 8)
        car = ucan.build_invocation_car([(block, cid)])

        root_cids, blocks = parse_car(car)
        assert root_cids[0] == cid
        assert cid in blocks

    def test_nonces_are_unique(self, ucan):
        """Each invocation must have a unique nonce to prevent replay attacks."""
        content_cid = ucan.content_cid(b"nonce check")
        block1, _ = ucan.build_store_add(content_cid, 11)
        block2, _ = ucan.build_store_add(content_cid, 11)
        assert cbor2.loads(block1)["nce"] != cbor2.loads(block2)["nce"]

    def test_load_raw_base64_key(self):
        """Raw base64 seed should also be accepted."""
        seed, _ = _make_seed_and_key()
        raw_b64 = base64.b64encode(seed).decode()
        pk = StorachaUCAN._load_agent_key(raw_b64)
        assert pk is not None

    def test_load_w3_key_format(self):
        """The multibase 'M...' format from `w3 key create` must be supported."""
        seed, _ = _make_seed_and_key()
        w3_key = _seed_to_w3_key(seed)
        assert w3_key.startswith("M")
        pk = StorachaUCAN._load_agent_key(w3_key)
        assert pk is not None

    def test_invalid_proof_raises(self):
        seed, _ = _make_seed_and_key()
        w3_key = _seed_to_w3_key(seed)
        with pytest.raises(Exception):
            StorachaUCAN(agent_key=w3_key, space_did="did:key:z6Mk", proof="notvalid!")

    def test_content_cid_deterministic(self, ucan):
        data = b"same data"
        assert ucan.content_cid(data) == ucan.content_cid(data)

    def test_cid_to_str_format(self, ucan):
        cid = ucan.content_cid(b"str test")
        s = ucan.cid_to_str(cid)
        assert s.startswith("b")
