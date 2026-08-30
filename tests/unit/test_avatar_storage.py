"""Unit tests for avatar storage in AsyncFileStore"""

import pytest
import json
from kestrel_sovereign.storage.async_file_store import AsyncFileStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
from kestrel_sovereign.storage.db import TransactionError


@pytest.fixture
async def file_store(tmp_path):
    """Create file store with temp database"""
    db = await AsyncDatabase.sqlite(str(tmp_path / "test.db"))
    store = AsyncFileStore(db)
    yield store
    await db.close()


@pytest.fixture
async def file_store_with_agent(tmp_path):
    """Create file store with a pre-existing agent node"""
    db = await AsyncDatabase.sqlite(str(tmp_path / "test_agent.db"))
    store = AsyncFileStore(db)

    # Create an agent node
    agent_id = "did:pkh:eip155:1:0xTestAgent"
    await db.execute_commit("""
        INSERT INTO graph_nodes (node_id, node_type, label, properties)
        VALUES (?, 'agent', 'Test Agent', ?)
    """, (agent_id, json.dumps({"created_at": "2025-01-01T00:00:00Z"})))

    yield store, agent_id
    await db.close()


class TestAvatarStorage:
    """Test avatar storage methods"""

    @pytest.mark.asyncio
    async def test_store_avatar_creates_file_and_graph_node(self, file_store):
        """Avatar storage creates file entry and graph relationships"""
        agent_id = "did:pkh:eip155:1:0x1234"
        image_data = b"fake_image_data_jpeg"

        content_hash = await file_store.store_avatar(
            image_data=image_data,
            agent_id=agent_id,
            avatar_type="primary",
            source_url="https://replicate.delivery/test.jpg"
        )

        # Verify file stored
        assert content_hash is not None
        assert len(content_hash) == 64  # SHA256 hex

        # Verify can retrieve
        retrieved = await file_store.retrieve_file(content_hash)
        assert retrieved == image_data

        # Verify graph node created
        metadata = await file_store.get_file_metadata(content_hash)
        assert metadata["type"] == "avatar"
        assert metadata["agent_id"] == agent_id

    @pytest.mark.asyncio
    async def test_store_avatar_creates_graph_edge(self, file_store):
        """Avatar storage creates graph edge linking agent to avatar"""
        agent_id = "did:pkh:eip155:1:0xEdgeTest"
        image_data = b"edge_test_image"

        content_hash = await file_store.store_avatar(
            image_data=image_data,
            agent_id=agent_id,
            avatar_type="primary"
        )

        # Verify edge created
        row = await file_store.db.fetchone("""
            SELECT source_id, target_id, label, properties
            FROM graph_edges
            WHERE source_id = ? AND label = 'has_avatar'
        """, (agent_id,))

        assert row is not None
        assert row[0] == agent_id  # source_id
        assert row[1] != content_hash  # tenant/type-namespaced graph id
        props = json.loads(row[3])
        assert props["avatar_type"] == "primary"
        avatar = await file_store.db.fetchone(
            "SELECT properties FROM graph_nodes WHERE node_id = ?",
            (row[1],),
        )
        assert json.loads(avatar[0])["hash"] == content_hash

    @pytest.mark.asyncio
    async def test_store_avatar_updates_agent_node_property(self, file_store_with_agent):
        """Primary avatar stores hash on agent node properties"""
        store, agent_id = file_store_with_agent
        image_data = b"agent_avatar_image"

        content_hash = await store.store_avatar(
            image_data=image_data,
            agent_id=agent_id,
            avatar_type="primary"
        )

        # Verify agent node has avatar_hash property
        row = await store.db.fetchone(
            "SELECT properties FROM graph_nodes WHERE node_id = ?",
            (agent_id,)
        )
        assert row is not None
        properties = json.loads(row[0])
        assert properties.get("avatar_hash") == content_hash

    @pytest.mark.asyncio
    async def test_get_agent_avatar_returns_image_bytes(self, file_store):
        """get_agent_avatar returns the stored image bytes"""
        agent_id = "did:pkh:eip155:1:0xGetTest"
        image_data = b"retrievable_image_data"

        await file_store.store_avatar(image_data, agent_id, "primary")

        # Retrieve via method
        avatar = await file_store.get_agent_avatar(agent_id, "primary")
        assert avatar == image_data

    @pytest.mark.asyncio
    async def test_get_agent_avatar_returns_latest(self, file_store):
        """get_agent_avatar returns most recent avatar when multiple exist"""
        agent_id = "did:pkh:eip155:1:0x5678"

        # Store first avatar
        await file_store.store_avatar(b"avatar_v1", agent_id, "primary")

        # Store second avatar (should replace)
        await file_store.store_avatar(b"avatar_v2", agent_id, "primary")

        # Should get latest
        avatar = await file_store.get_agent_avatar(agent_id, "primary")
        assert avatar == b"avatar_v2"

    @pytest.mark.asyncio
    async def test_get_agent_avatar_hash_returns_hash(self, file_store):
        """get_agent_avatar_hash returns content hash for URL generation"""
        agent_id = "did:pkh:eip155:1:0xabcd"
        image_data = b"test_avatar_image"

        stored_hash = await file_store.store_avatar(image_data, agent_id, "primary")
        retrieved_hash = await file_store.get_agent_avatar_hash(agent_id, "primary")

        assert retrieved_hash == stored_hash

    @pytest.mark.asyncio
    async def test_get_agent_avatar_hash_from_agent_node(self, file_store_with_agent):
        """get_agent_avatar_hash checks agent node property first"""
        store, agent_id = file_store_with_agent
        image_data = b"node_avatar_image"

        stored_hash = await store.store_avatar(image_data, agent_id, "primary")

        # Method should find hash from agent node properties
        retrieved_hash = await store.get_agent_avatar_hash(agent_id, "primary")
        assert retrieved_hash == stored_hash

    @pytest.mark.asyncio
    async def test_avatar_not_found_returns_none(self, file_store):
        """Non-existent avatar returns None"""
        avatar = await file_store.get_agent_avatar("nonexistent_agent", "primary")
        assert avatar is None

        avatar_hash = await file_store.get_agent_avatar_hash("nonexistent_agent", "primary")
        assert avatar_hash is None

    @pytest.mark.asyncio
    async def test_multiple_avatar_types(self, file_store):
        """Different avatar types stored separately"""
        agent_id = "did:pkh:eip155:1:0xmulti"

        await file_store.store_avatar(b"primary_avatar", agent_id, "primary")
        await file_store.store_avatar(b"selfie_avatar", agent_id, "selfie")
        await file_store.store_avatar(b"thumbnail_avatar", agent_id, "thumbnail")

        assert await file_store.get_agent_avatar(agent_id, "primary") == b"primary_avatar"
        assert await file_store.get_agent_avatar(agent_id, "selfie") == b"selfie_avatar"
        assert await file_store.get_agent_avatar(agent_id, "thumbnail") == b"thumbnail_avatar"

    @pytest.mark.asyncio
    async def test_avatar_deduplication(self, file_store):
        """Identical images produce same hash (content-addressed)"""
        agent_id = "did:pkh:eip155:1:0xdedup"
        same_image = b"identical_image_data"

        hash1 = await file_store.store_avatar(same_image, agent_id, "primary")
        hash2 = await file_store.store_avatar(same_image, agent_id, "variant_1")

        assert hash1 == hash2  # Same content = same hash
        assert await file_store.get_agent_avatar(agent_id, "primary") == same_image
        assert await file_store.get_agent_avatar(agent_id, "variant_1") == same_image

    @pytest.mark.asyncio
    async def test_identical_avatar_bytes_keep_tenant_graph_metadata_separate(
        self, tmp_path
    ):
        db = await AsyncDatabase.sqlite(str(tmp_path / "shared-avatar.db"))
        agent_a = "did:test:avatar-a"
        agent_b = "did:test:avatar-b"
        store_a = AsyncFileStore(db, agent_id=agent_a)
        store_b = AsyncFileStore(db, agent_id=agent_b)
        image = b"same-avatar-bytes"
        try:
            hash_a = await store_a.store_avatar(image, agent_a, "primary")
            hash_b = await store_b.store_avatar(image, agent_b, "primary")

            assert hash_a == hash_b
            assert await store_a.get_agent_avatar(agent_a, "primary") == image
            assert await store_b.get_agent_avatar(agent_b, "primary") == image
            nodes = await db.fetchall(
                "SELECT node_id, properties FROM graph_nodes "
                "WHERE node_type = 'avatar' ORDER BY node_id"
            )
            assert len(nodes) == 2
            assert nodes[0][0] != nodes[1][0]
            assert {json.loads(row[1])["agent_id"] for row in nodes} == {
                agent_a,
                agent_b,
            }
            assert await db.fetchall(
                "SELECT agent_id FROM file_owners "
                "WHERE content_hash = ? ORDER BY agent_id",
                (hash_a,),
            ) == [(agent_a,), (agent_b,)]
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_store_avatar_rolls_back_file_reference_on_graph_failure(
        self, file_store, monkeypatch
    ):
        async def fail_graph_write(*_args, **_kwargs):
            raise RuntimeError("injected avatar graph failure")

        monkeypatch.setattr(AsyncGraphStore, "add_node", fail_graph_write)
        with pytest.raises(TransactionError, match="injected avatar graph failure"):
            await file_store.store_avatar(
                b"must-rollback", "did:test:avatar-rollback", "primary"
            )

        assert await file_store.db.fetchone("SELECT COUNT(*) FROM files") == (0,)
        assert await file_store.db.fetchone(
            "SELECT COUNT(*) FROM file_owners"
        ) == (0,)
        assert await file_store.db.fetchone(
            "SELECT COUNT(*) FROM graph_node_owners"
        ) == (0,)

    @pytest.mark.asyncio
    async def test_store_avatar_rejects_foreign_node_at_provisional_agent_id(
        self, file_store
    ):
        """Avatar bootstrap cannot claim a graph id controlled by another tenant."""

        agent_id = "did:test:avatar-victim"
        attacker_id = "did:test:avatar-attacker"
        await file_store.db.execute_commit(
            "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
            "VALUES (?, 'skill', 'Squat', ?)",
            (agent_id, json.dumps({"agent_id": attacker_id})),
        )
        await file_store.db.execute_commit(
            "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
            (agent_id, attacker_id),
        )

        with pytest.raises(TransactionError, match="collides"):
            await file_store.store_avatar(b"must-not-persist", agent_id, "primary")

        assert await file_store.db.fetchone("SELECT COUNT(*) FROM files") == (0,)
        assert await file_store.db.fetchall(
            "SELECT agent_id FROM graph_node_owners WHERE node_id = ?",
            (agent_id,),
        ) == [(attacker_id,)]
        assert await file_store.db.fetchone(
            "SELECT 1 FROM graph_nodes WHERE node_type = 'avatar'"
        ) is None

    @pytest.mark.asyncio
    async def test_avatar_metadata_includes_source_url(self, file_store):
        """Avatar metadata includes original source URL"""
        agent_id = "did:pkh:eip155:1:0xsource"
        source_url = "https://replicate.delivery/pbxt/abc123/output.jpg"

        content_hash = await file_store.store_avatar(
            b"image_with_source",
            agent_id,
            "primary",
            source_url=source_url
        )

        metadata = await file_store.get_file_metadata(content_hash)
        assert metadata["source_url"] == source_url

    @pytest.mark.asyncio
    async def test_avatar_metadata_includes_mime_type(self, file_store):
        """Avatar metadata includes MIME type"""
        agent_id = "did:pkh:eip155:1:0xmime"

        content_hash = await file_store.store_avatar(
            b"jpeg_image_data",
            agent_id,
            "primary"
        )

        metadata = await file_store.get_file_metadata(content_hash)
        assert metadata["mime_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_avatar_with_variant_types(self, file_store):
        """Non-primary avatar types don't update agent node"""
        agent_id = "did:pkh:eip155:1:0xvariant"

        # Create agent node first
        await file_store.db.execute_commit("""
            INSERT INTO graph_nodes (node_id, node_type, label, properties)
            VALUES (?, 'agent', 'Variant Agent', ?)
        """, (agent_id, json.dumps({})))

        # Store variant (not primary)
        await file_store.store_avatar(b"variant_image", agent_id, "variant_1")

        # Agent node should NOT have avatar_hash (only primary updates it)
        row = await file_store.db.fetchone(
            "SELECT properties FROM graph_nodes WHERE node_id = ?",
            (agent_id,)
        )
        properties = json.loads(row[0])
        assert "avatar_hash" not in properties
