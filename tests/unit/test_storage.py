import pytest
import pytest_asyncio
import os
from pathlib import Path
import hashlib
import tempfile

# Adjust the python path to include the root of the project
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kestrel_sovereign.storage import AsyncStorage, GraphNode

@pytest_asyncio.fixture
async def storage(temp_dir):
    """Provides an AsyncStorage instance in a temporary directory for each test."""
    db_path = str(temp_dir / "test_storage.db")
    storage_instance = AsyncStorage(db_path=db_path)
    await storage_instance.initialize()
    yield storage_instance
    await storage_instance.close()

@pytest.fixture
def test_file(tmpdir):
    """Creates a temporary test file."""
    test_file_path = tmpdir.join("test_doc.txt")
    content = "This is a test document about Kestrel, an AI for privacy."
    test_file_path.write(content)
    return test_file_path, content

@pytest.mark.asyncio
async def test_file_storage(storage, test_file):
    """Test storing and retrieving a file."""
    test_file_path, content_str = test_file
    content_bytes = content_str.encode('utf-8')
    metadata = {"source": "unit_test"}

    content_hash = await storage.store_file(content_bytes, os.path.basename(test_file_path), metadata=metadata)

    # Verify hash
    expected_hash = hashlib.sha256(content_bytes).hexdigest()
    assert content_hash == expected_hash

    # Verify retrieval
    retrieved_content = await storage.retrieve_file(content_hash)
    assert retrieved_content == content_bytes

    # Verify metadata
    file_meta = await storage.get_file_metadata(content_hash)
    assert file_meta['source'] == 'unit_test'

@pytest.mark.asyncio
async def test_knowledge_graph(storage):
    """Test adding and querying nodes and edges."""
    # Add nodes
    node1 = GraphNode(node_id="person_1", node_type="person", label="UncleSaurus", properties={"role": "creator"})
    node2 = GraphNode(node_id="agent_1", node_type="agent", label="Kestrel", properties={"version": "1.0"})
    await storage.add_node(node1)
    await storage.add_node(node2)

    # Add edge
    await storage.add_edge("person_1", "agent_1", "created")

    # Query graph
    retrieved_node = await storage.get_node("person_1")
    assert retrieved_node.label == "UncleSaurus"

@pytest.mark.asyncio
async def test_rag_pipeline(storage, test_file):
    """Test chunking a document and searching the chunks."""
    test_file_path, content_str = test_file
    content_bytes = content_str.encode('utf-8')
    content_hash = await storage.store_file(content_bytes, os.path.basename(test_file_path))
    await storage.chunk_document(content_hash)

    # Search chunks
    search_results = await storage.search_chunks("privacy")
    assert len(search_results) >= 1
    assert "Kestrel, an AI for privacy" in search_results[0]['content']

@pytest.mark.asyncio
async def test_conversation_history(storage):
    """Test adding and retrieving conversation history."""
    await storage.add_conversation("user", "Hello Kestrel")
    await storage.add_conversation("assistant", "Hello! How may I help you?")

    history = await storage.get_conversation_history(limit=5)
    assert len(history) == 2
    assert history[0]['role'] == 'user'
    assert history[1]['content'] == 'Hello! How may I help you?'

@pytest.mark.asyncio
async def test_backup_local_only(storage):
    """Test creating a local-only backup blob and recording it as an artifact without external dependencies."""
    # Create a tiny conversation to ensure DB has content
    await storage.add_conversation("user", "backup me")
    blob = await storage.create_backup_blob(include_db=True)
    assert isinstance(blob, (bytes, bytearray))
    assert len(blob) > 0

    # Fake a minimal StorageResult-like object
    class _FakeResult:
        def __init__(self, content_hash: str):
            self.content_hash = content_hash
            self.storage_tier = type("T", (), {"value": "local"})()
            self.ipfs_cid = None
            self.filecoin_deal_id = None
            self.encrypted = False
            self.encryption_key_hash = None

    # Use a deterministic hash for the fake result id
    import hashlib
    ch = hashlib.sha256(blob).hexdigest()
    fake = _FakeResult(ch)
    agent_id = "did:pkh:eip155:1:0xTEST"
    node_id = await storage.record_backup_artifact(agent_id, fake)
    assert node_id == ch
    # Ensure node is queryable
    node = await storage.get_node(ch)
    assert node is not None
    assert node.node_type == "backup_artifact"
