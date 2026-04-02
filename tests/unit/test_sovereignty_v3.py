"""
Unit tests for V3 Sovereignty: AssetCollector, manifest v3, and CAR packaging.

Covers issues #367 (AssetCollector + manifest v3) and #368 (CAR packaging).
"""

import hashlib
import json
import pytest

try:
    import cbor2 as _cbor2
    _has_cbor2 = True
except ImportError:
    _has_cbor2 = False

_skip_no_cbor2 = pytest.mark.skipif(not _has_cbor2, reason="cbor2 not installed (wallet extras)")

from kestrel_sovereign.storage.sovereign_adapter import (
    AssetCollector,
    AssetDescriptor,
    AssetMetadata,
    ConvergentEncryptor,
    RootManifest,
    ShardMetadata,
    SovereignStorageAdapter,
    MANIFEST_VERSION,
)
from kestrel_sovereign.storage.car_builder import CARBuilder, CARReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubAssetCollector(AssetCollector):
    """Test collector that returns pre-built descriptors."""

    def __init__(self, descriptors):
        self._descriptors = descriptors

    async def collect_assets(self, agent_did):
        return self._descriptors


# ---------------------------------------------------------------------------
# Dataclass / protocol tests (#367)
# ---------------------------------------------------------------------------

class TestAssetDescriptor:
    def test_inline_asset(self):
        data = b"avatar-png-bytes"
        desc = AssetDescriptor(
            asset_type="avatar",
            asset_key="avatar_main",
            content_hash=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            data=data,
        )
        assert desc.ipfs_cid is None
        assert desc.encrypted is False
        assert desc.data == data

    def test_external_ref_asset(self):
        desc = AssetDescriptor(
            asset_type="lora_weights",
            asset_key="lora_v2",
            content_hash="abc123",
            size_bytes=170_000_000,
            ipfs_cid="bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
        )
        assert desc.data is None
        assert desc.ipfs_cid is not None


class TestAssetMetadata:
    def test_roundtrip_dict(self):
        meta = AssetMetadata(
            asset_type="selfie",
            asset_key="selfie_001",
            cid="bafytest",
            content_hash="deadbeef",
            size_bytes=4096,
            metadata={"taken_at": "2026-03-28"},
        )
        from dataclasses import asdict
        d = asdict(meta)
        restored = AssetMetadata(**d)
        assert restored == meta


class TestManifestV3:
    def test_version_is_3(self):
        assert MANIFEST_VERSION == "3.0"

    def test_manifest_with_assets(self):
        m = RootManifest(
            version="3.0",
            timestamp="2026-03-28T00:00:00Z",
            agent_did="did:test:1",
            shards=[],
            assets=[
                AssetMetadata(
                    asset_type="avatar",
                    asset_key="main",
                    cid="bafyavatar",
                    content_hash="aaa",
                    size_bytes=100,
                )
            ],
        )
        assert len(m.assets) == 1
        assert m.assets[0].asset_type == "avatar"

    def test_manifest_defaults_empty_assets(self):
        m = RootManifest(
            version="3.0",
            timestamp="2026-03-28T00:00:00Z",
            agent_did="did:test:1",
            shards=[],
        )
        assert m.assets == []


class TestAssetCollectorProtocol:
    @pytest.mark.asyncio
    async def test_stub_collector(self):
        data = b"test-data"
        collector = StubAssetCollector([
            AssetDescriptor(
                asset_type="personality",
                asset_key="core",
                content_hash=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                data=data,
            )
        ])
        assets = await collector.collect_assets("did:test:1")
        assert len(assets) == 1
        assert assets[0].asset_key == "core"


# ---------------------------------------------------------------------------
# CARBuilder external ref (#368)
# ---------------------------------------------------------------------------

@_skip_no_cbor2
class TestCARExternalRef:
    def test_add_external_ref(self):
        builder = CARBuilder()
        # Add a real block first so we have a valid CID to reference
        inner_cid = builder.add_raw_block(b"some data")
        ref_cid = builder.add_external_ref(inner_cid, ref_type="lora_weights")

        # The ref CID should be different from the inner CID (it's a dag-cbor node)
        assert ref_cid != inner_cid

        # Build and parse
        builder.set_root(ref_cid)
        car_bytes = builder.build()
        reader = CARReader(car_bytes)
        assert reader.verify()

        # The link node should contain the reference
        link_node = reader.get_dag_cbor_block(ref_cid)
        assert link_node is not None
        assert link_node["type"] == "lora_weights"


# ---------------------------------------------------------------------------
# CAR roundtrip with encryption (#368)
# ---------------------------------------------------------------------------

@_skip_no_cbor2
class TestCAREncryptedRoundtrip:
    def test_encrypt_pack_unpack_decrypt(self):
        """Verify that encrypted shards survive CAR packing and unpacking."""
        encryptor = ConvergentEncryptor("test-secret")
        builder = CARBuilder()

        # Encrypt two shards
        shard1 = json.dumps({"msg": "hello"}).encode()
        shard2 = json.dumps({"msg": "world"}).encode()
        cipher1, key1 = encryptor.encrypt_with_nonce_prefix(shard1)
        cipher2, key2 = encryptor.encrypt_with_nonce_prefix(shard2)

        cid1 = builder.add_raw_block(cipher1)
        cid2 = builder.add_raw_block(cipher2)

        manifest = {"shards": [cid1, cid2], "version": "3.0"}
        root_cid = builder.add_dag_cbor_block(manifest)
        builder.set_root(root_cid)
        car_bytes = builder.build()

        # Parse
        reader = CARReader(car_bytes)
        assert reader.verify()
        assert reader.block_count == 3  # 2 shards + 1 manifest

        # Decrypt
        restored1 = encryptor.decrypt(reader.get_block(cid1), key1)
        restored2 = encryptor.decrypt(reader.get_block(cid2), key2)
        assert json.loads(restored1) == {"msg": "hello"}
        assert json.loads(restored2) == {"msg": "world"}


@_skip_no_cbor2
class TestCARWithAssets:
    def test_inline_and_external_assets(self):
        """CAR can hold inline asset blocks and external references."""
        builder = CARBuilder()
        encryptor = ConvergentEncryptor("secret")

        # Inline asset (encrypted)
        avatar_data = b"PNG-AVATAR-BYTES"
        avatar_cipher, avatar_key = encryptor.encrypt_with_nonce_prefix(avatar_data)
        avatar_cid = builder.add_raw_block(avatar_cipher)

        # External reference (LoRA already on IPFS)
        lora_external_cid = builder.add_raw_block(b"placeholder-for-cid-generation")
        ref_cid = builder.add_external_ref(lora_external_cid, ref_type="lora_weights")

        # Manifest
        manifest = {
            "version": "3.0",
            "assets": [
                {"cid": avatar_cid, "type": "avatar", "encrypted": True},
                {"cid": ref_cid, "type": "lora_weights", "encrypted": False},
            ],
        }
        root = builder.add_dag_cbor_block(manifest)
        builder.set_root(root)
        car_bytes = builder.build()

        reader = CARReader(car_bytes)
        assert reader.verify()

        # Inline asset can be decrypted
        restored_avatar = encryptor.decrypt(reader.get_block(avatar_cid), avatar_key)
        assert restored_avatar == avatar_data

        # External ref node exists
        ref_node = reader.get_dag_cbor_block(ref_cid)
        assert ref_node["type"] == "lora_weights"
