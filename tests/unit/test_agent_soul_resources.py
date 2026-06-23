import pytest

from kestrel_sovereign.agent.context_builder import ContextBuilder
from kestrel_sovereign.storage import AsyncStorage, SOUL_MARKDOWN_RESOURCE_TYPE


TEST_AGENT_ID = "did:pkh:eip155:1:0xfeedfacefeedfacefeedfacefeedfacefeedface"


@pytest.fixture(autouse=True)
def data_key(monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-soul-resource-key")


async def _storage(tmp_path):
    storage = AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=TEST_AGENT_ID)
    await storage.initialize()
    return storage


@pytest.mark.asyncio
async def test_soul_resource_round_trip_is_encrypted(tmp_path):
    storage = await _storage(tmp_path)
    try:
        body = "# SOUL.md\n\nprivate identity body"
        resource = await storage.promote_soul_seed(
            body,
            created_by=TEST_AGENT_ID,
            source="agent_data/Emma/SOUL.md",
        )

        current = await storage.get_current_agent_resource()
        assert current is not None
        assert current.content == body
        assert current.resource_type == SOUL_MARKDOWN_RESOURCE_TYPE
        assert current.version == 1
        assert current.provenance["created_by"] == TEST_AGENT_ID
        assert current.provenance["source"] == "agent_data/Emma/SOUL.md"

        row = await storage.db.fetchone(
            """
            SELECT content_ciphertext, content_hash, public_metadata
            FROM agent_identity_resources
            WHERE id = ?
            """,
            (resource.id,),
        )
        ciphertext = row[0]
        if isinstance(ciphertext, memoryview):
            ciphertext = ciphertext.tobytes()
        if isinstance(ciphertext, str):
            ciphertext_bytes = ciphertext.encode("utf-8")
        else:
            ciphertext_bytes = bytes(ciphertext)
        assert b"private identity body" not in ciphertext_bytes
        assert row[1] == resource.content_hash
        assert "private identity body" not in row[2]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_soul_resource_versions_select_current_and_keep_provenance(tmp_path):
    storage = await _storage(tmp_path)
    try:
        first = await storage.promote_soul_seed(
            "first soul",
            created_by="bootstrap",
            source="default-template",
        )
        second = await storage.create_agent_resource_version(
            SOUL_MARKDOWN_RESOURCE_TYPE,
            "second soul",
            created_by=TEST_AGENT_ID,
            source="rename:/agent/SOUL.md",
            signature={"alg": "test", "value": "sig"},
            anchoring_metadata={"hash_pointer": "sha256:abc"},
        )

        assert first.resource_id == second.resource_id
        assert first.version == 1
        assert second.version == 2

        current = await storage.get_current_agent_resource()
        assert current.content == "second soul"
        assert current.version == 2
        assert current.provenance["source"] == "rename:/agent/SOUL.md"
        assert current.provenance["signature"]["value"] == "sig"

        public = await storage.get_agent_resource_public_metadata()
        assert public["resource_type"] == SOUL_MARKDOWN_RESOURCE_TYPE
        assert public["version"] == 2
        assert public["content_hash"] == second.content_hash
        assert public["anchoring"] == {"hash_pointer": "sha256:abc"}
        assert "second soul" not in str(public)
        assert "content" not in public

        node = await storage.get_node(f"{TEST_AGENT_ID}#soul")
        assert node is not None
        assert node.properties["private_body"] is True
        assert node.properties["content_hash"] == second.content_hash
        assert "second soul" not in str(node.properties)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_context_builder_loads_canonical_soul_over_seed(tmp_path):
    seed = tmp_path / "SOUL.md"
    seed.write_text("seed soul", encoding="utf-8")

    storage = await _storage(tmp_path)
    try:
        await storage.promote_soul_seed(
            "canonical soul",
            created_by=TEST_AGENT_ID,
            source=str(seed),
        )
        builder = ContextBuilder(storage, agent_data_path=str(tmp_path))
        assert builder._soul_content == "seed soul"

        loaded = await builder.load_canonical_soul_resource()
        assert loaded is True
        assert builder._soul_content == "canonical soul"
    finally:
        await storage.close()
