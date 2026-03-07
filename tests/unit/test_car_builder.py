"""Tests for CAR v1 builder and reader."""

import hashlib

import pytest

from kestrel_sovereign.storage.car_builder import (
    CARBuilder,
    CARReader,
    encode_varint,
    decode_varint,
    compute_raw_cid,
    compute_dag_cbor_cid,
    cid_bytes_to_string,
    cid_string_to_bytes,
    build_cid_bytes,
    CODEC_RAW,
    CODEC_DAG_CBOR,
    make_cid_link,
    _dag_cbor_encode,
)


class TestVarint:
    """Test varint (unsigned LEB128) encoding/decoding."""

    def test_encode_zero(self):
        assert encode_varint(0) == b"\x00"

    def test_encode_small(self):
        assert encode_varint(1) == b"\x01"
        assert encode_varint(127) == b"\x7f"

    def test_encode_two_bytes(self):
        assert encode_varint(128) == b"\x80\x01"
        assert encode_varint(300) == b"\xac\x02"

    def test_encode_large(self):
        # 16384 = 0x4000
        result = encode_varint(16384)
        assert len(result) == 3

    def test_roundtrip(self):
        for n in [0, 1, 127, 128, 255, 256, 300, 16384, 1000000]:
            encoded = encode_varint(n)
            decoded, end = decode_varint(encoded)
            assert decoded == n
            assert end == len(encoded)

    def test_decode_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            encode_varint(-1)


class TestCID:
    """Test CID computation and encoding."""

    def test_raw_cid_deterministic(self):
        data = b"hello world"
        cid1_bytes, cid1_str = compute_raw_cid(data)
        cid2_bytes, cid2_str = compute_raw_cid(data)
        assert cid1_bytes == cid2_bytes
        assert cid1_str == cid2_str

    def test_raw_cid_starts_with_b(self):
        _, cid_str = compute_raw_cid(b"test")
        assert cid_str.startswith("b")

    def test_different_data_different_cids(self):
        _, cid1 = compute_raw_cid(b"hello")
        _, cid2 = compute_raw_cid(b"world")
        assert cid1 != cid2

    def test_dag_cbor_cid_differs_from_raw(self):
        """Same data, different codecs → different CIDs."""
        data = b"test"
        raw_bytes = build_cid_bytes(CODEC_RAW, data)
        cbor_bytes = build_cid_bytes(CODEC_DAG_CBOR, data)
        assert raw_bytes != cbor_bytes

    def test_cid_string_roundtrip(self):
        cid_bytes, cid_str = compute_raw_cid(b"roundtrip test")
        recovered = cid_string_to_bytes(cid_str)
        assert recovered == cid_bytes

    def test_cid_string_to_bytes_rejects_bad_prefix(self):
        with pytest.raises(ValueError, match="multibase prefix"):
            cid_string_to_bytes("zNotBase32")


class TestCARBuilder:
    """Test CAR building."""

    def test_build_requires_root(self):
        builder = CARBuilder()
        builder.add_raw_block(b"data")
        with pytest.raises(ValueError, match="Root CID"):
            builder.build()

    def test_build_requires_blocks(self):
        builder = CARBuilder()
        with pytest.raises(ValueError, match="Root CID"):
            builder.build()

    def test_set_root_requires_existing_cid(self):
        builder = CARBuilder()
        with pytest.raises(ValueError, match="not found"):
            builder.set_root("bfakecidinvalid")

    def test_single_block(self):
        builder = CARBuilder()
        cid = builder.add_raw_block(b"hello world")
        builder.set_root(cid)
        car = builder.build()
        assert isinstance(car, bytes)
        assert len(car) > 0

    def test_multiple_blocks(self):
        builder = CARBuilder()
        cid1 = builder.add_raw_block(b"block one")
        cid2 = builder.add_raw_block(b"block two")
        manifest_cid = builder.add_dag_cbor_block({"blocks": [cid1, cid2]})
        builder.set_root(manifest_cid)
        car = builder.build()
        assert builder.block_count == 3

    def test_deduplicates_identical_blocks(self):
        builder = CARBuilder()
        cid1 = builder.add_raw_block(b"same data")
        cid2 = builder.add_raw_block(b"same data")
        assert cid1 == cid2
        assert builder.block_count == 1


class TestCARReader:
    """Test CAR reading and verification."""

    def _build_simple_car(self) -> bytes:
        builder = CARBuilder()
        cid = builder.add_raw_block(b"hello world")
        builder.set_root(cid)
        return builder.build()

    def test_parse_simple(self):
        car = self._build_simple_car()
        reader = CARReader(car)
        assert reader.block_count == 1
        assert reader.root_cid

    def test_get_block(self):
        builder = CARBuilder()
        data = b"test data 12345"
        cid = builder.add_raw_block(data)
        builder.set_root(cid)
        car = builder.build()

        reader = CARReader(car)
        assert reader.get_block(cid) == data

    def test_get_nonexistent_block(self):
        car = self._build_simple_car()
        reader = CARReader(car)
        assert reader.get_block("bnonexistent") is None

    def test_list_cids(self):
        builder = CARBuilder()
        cid1 = builder.add_raw_block(b"one")
        cid2 = builder.add_raw_block(b"two")
        manifest_cid = builder.add_dag_cbor_block({"parts": [cid1, cid2]})
        builder.set_root(manifest_cid)
        car = builder.build()

        reader = CARReader(car)
        cids = reader.list_cids()
        assert len(cids) == 3
        assert cid1 in cids
        assert cid2 in cids
        assert manifest_cid in cids

    def test_verify_valid(self):
        car = self._build_simple_car()
        reader = CARReader(car)
        assert reader.verify() is True

    def test_dag_cbor_block_decode(self):
        builder = CARBuilder()
        obj = {"key": "value", "num": 42}
        cid = builder.add_dag_cbor_block(obj)
        builder.set_root(cid)
        car = builder.build()

        reader = CARReader(car)
        decoded = reader.get_dag_cbor_block(cid)
        assert decoded == obj


class TestCARRoundTrip:
    """Test full build → parse → extract round-trips."""

    def test_single_block_roundtrip(self):
        data = b"sovereignty export shard data"
        builder = CARBuilder()
        cid = builder.add_raw_block(data)
        builder.set_root(cid)
        car = builder.build()

        reader = CARReader(car)
        assert reader.root_cid == cid
        assert reader.get_block(cid) == data
        assert reader.verify()

    def test_sovereignty_export_pattern(self):
        """Test the typical sovereignty export: shards + keyring + manifest."""
        builder = CARBuilder()

        # Add encrypted shards
        shard1 = b"encrypted shard 2025-11 data" * 100
        shard2 = b"encrypted shard 2025-12 data" * 100
        cid1 = builder.add_raw_block(shard1)
        cid2 = builder.add_raw_block(shard2)

        # Add keyring
        keyring = b"encrypted keyring data"
        keyring_cid = builder.add_raw_block(keyring)

        # Add manifest pointing to all parts
        manifest = {
            "version": "2.0",
            "agent_did": "did:pkh:eip155:1:0xabc",
            "shards": [
                {"shard_id": "conv_2025-11", "cid": cid1},
                {"shard_id": "conv_2025-12", "cid": cid2},
            ],
            "keyring_cid": keyring_cid,
        }
        manifest_cid = builder.add_dag_cbor_block(manifest)
        builder.set_root(manifest_cid)

        # Build and verify
        car = builder.build()
        assert builder.block_count == 4

        # Parse back
        reader = CARReader(car)
        assert reader.root_cid == manifest_cid
        assert reader.block_count == 4
        assert reader.verify()

        # Extract manifest
        parsed_manifest = reader.get_dag_cbor_block(manifest_cid)
        assert parsed_manifest["version"] == "2.0"
        assert parsed_manifest["agent_did"] == "did:pkh:eip155:1:0xabc"
        assert len(parsed_manifest["shards"]) == 2

        # Extract shards
        assert reader.get_block(cid1) == shard1
        assert reader.get_block(cid2) == shard2
        assert reader.get_block(keyring_cid) == keyring

    def test_empty_block(self):
        builder = CARBuilder()
        cid = builder.add_raw_block(b"")
        builder.set_root(cid)
        car = builder.build()

        reader = CARReader(car)
        assert reader.get_block(cid) == b""
        assert reader.verify()

    def test_large_block(self):
        data = bytes(range(256)) * 1000  # 256KB
        builder = CARBuilder()
        cid = builder.add_raw_block(data)
        builder.set_root(cid)
        car = builder.build()

        reader = CARReader(car)
        assert reader.get_block(cid) == data
        assert reader.verify()
